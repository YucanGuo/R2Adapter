# -*- coding: utf-8 -*-
"""
Router-integrated HippoRAG code file
Decides query retrieval method based on trained router model:
- If router predicts graph (>threshold), use HippoRAG for graph retrieval
- If router predicts passage (<=threshold), use StandardRAG for dense retrieval
"""

import os
import json
import torch
import numpy as np
from typing import List, Union, Tuple, Dict, Any
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
import logging
from tqdm import tqdm
from datetime import datetime
from rewriter import QueryRewriter
import time
import sys


# Add HippoRAG to path
sys.path.append('HippoRAG/src')

# Import HippoRAG related modules
from hipporag.HippoRAG import HippoRAG
from hipporag.StandardRAG import StandardRAG
from hipporag.utils.config_utils import BaseConfig
from hipporag.utils.misc_utils import string_to_bool

logger = logging.getLogger(__name__)


def get_gold_docs(samples: List, dataset_name: str = None) -> List:
    """Extract gold documents from samples for evaluation"""
    gold_docs = []
    for sample in samples:
        if 'supporting_facts' in sample:  # hotpotqa, 2wikimultihopqa
            gold_title = set([item[0] for item in sample['supporting_facts']])
            gold_title_and_content_list = [item for item in sample['context'] if item[0] in gold_title]
            if dataset_name.startswith('hotpotqa'):
                gold_doc = [item[0] + '\n' + ''.join(item[1]) for item in gold_title_and_content_list]
            else:
                gold_doc = [item[0] + '\n' + ' '.join(item[1]) for item in gold_title_and_content_list]
        elif 'contexts' in sample:
            gold_doc = [item['title'] + '\n' + item['text'] for item in sample['contexts'] if item['is_supporting']]
        else:
            assert 'paragraphs' in sample, "`paragraphs` should be in sample, or consider the setting not to evaluate retrieval"
            gold_paragraphs = []
            for item in sample['paragraphs']:
                if 'is_supporting' in item and item['is_supporting'] is False:
                    continue
                gold_paragraphs.append(item)
            gold_doc = [item['title'] + '\n' + (item['text'] if 'text' in item else item['paragraph_text']) for item in gold_paragraphs]

        gold_doc = list(set(gold_doc))
        gold_docs.append(gold_doc)
    return gold_docs


def get_gold_answers(samples):
    """Extract gold answers from samples for evaluation"""
    gold_answers = []
    for sample_idx in range(len(samples)):
        gold_ans = None
        sample = samples[sample_idx]

        if 'answer' in sample or 'gold_ans' in sample:
            gold_ans = sample['answer'] if 'answer' in sample else sample['gold_ans']
        elif 'reference' in sample:
            gold_ans = sample['reference']
        elif 'obj' in sample:
            gold_ans = set(
                [sample['obj']] + [sample['possible_answers']] + [sample['o_wiki_title']] + [sample['o_aliases']])
            gold_ans = list(gold_ans)
        assert gold_ans is not None
        if isinstance(gold_ans, str):
            gold_ans = [gold_ans]
        assert isinstance(gold_ans, list)
        gold_ans = set(gold_ans)
        if 'answer_aliases' in sample:
            gold_ans.update(sample['answer_aliases'])

        gold_answers.append(gold_ans)

    return gold_answers

class RouterIntegratedHippoRAG:
    """
    Router-integrated HippoRAG class
    Selects retrieval strategy based on router model predictions
    """
    
    def __init__(self, 
                 router_model_path: str,
                 router_threshold: float = 0.5,
                 hipporag_config: BaseConfig = None,
                 enable_query_rewriting: bool = True,
                 low_probability_threshold: float = 0.7,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 log_file: str = None):
        """
        Initialize Router-integrated HippoRAG
        
        Args:
            router_model_path: Path to trained router model
            router_threshold: Router prediction threshold, above which uses HippoRAG, otherwise StandardRAG
            hipporag_config: HippoRAG configuration
            device: Computing device
            log_file: Log file path for output
        """
        self.device = device
        self.router_threshold = 0.0 # router_threshold
        self.log_file = log_file
        
        # Setup logging
        self._setup_logging()
        
        # Load router model
        self._load_router_model(router_model_path)
        if enable_query_rewriting:
            self.query_rewriter = QueryRewriter(
                llm_base_url=hipporag_config.llm_base_url,
                llm_name=hipporag_config.llm_name,
                low_probability_threshold=low_probability_threshold
            )
        else:
            self.query_rewriter = None
        
        # Initialize HippoRAG and StandardRAG
        self.hipporag = HippoRAG(global_config=hipporag_config) if hipporag_config else None
        self.standard_rag = StandardRAG(global_config=hipporag_config) if hipporag_config else None
        
        # Statistics
        self.router_stats = {
            "total_queries": 0,
            "graph_queries": 0,
            "passage_queries": 0,
            "rewritten_queries": 0,
            "router_accuracy": 0.0
        }
        
        # Performance metrics
        self.performance_metrics = {
            "total_retrieval_time": 0.0,
            "total_qa_time": 0.0,
            "retrieval_recall_at_k": {},
            "qa_em_scores": [],
            "qa_f1_scores": []
        }
        
        # Detailed statistics for graph and passage strategies
        self.detailed_stats = {
            "graph": {
                "query_count": 0,
                "retrieval_time": 0.0,
                "qa_time": 0.0,
                "retrieval_recall_at_k": {},
                "qa_em_score": None,  # Single aggregated EM score for all graph queries
                "qa_f1_score": None,  # Single aggregated F1 score for all graph queries
                "avg_retrieval_time_per_query": 0.0,
                "avg_qa_time_per_query": 0.0,
                "avg_em_score": 0.0,
                "avg_f1_score": 0.0
            },
            "passage": {
                "query_count": 0,
                "retrieval_time": 0.0,
                "qa_time": 0.0,
                "retrieval_recall_at_k": {},
                "qa_em_score": None,  # Single aggregated EM score for all passage queries
                "qa_f1_score": None,  # Single aggregated F1 score for all passage queries
                "avg_retrieval_time_per_query": 0.0,
                "avg_qa_time_per_query": 0.0,
                "avg_em_score": 0.0,
                "avg_f1_score": 0.0
            }
        }
    
    def _setup_logging(self):
        """Setup logging configuration"""
        if self.log_file:
            # Create log directory if it doesn't exist
            log_dir = os.path.dirname(self.log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            # Configure file handler
            file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            
            # Configure formatter
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            
            # Add file handler to logger
            logger.addHandler(file_handler)
            
            logger.info(f"Logging configured to file: {self.log_file}")
    
    def _load_router_model(self, model_path: str):
        """Load trained router model"""
        try:
            # Load tokenizer
            self.router_tokenizer = DebertaV2Tokenizer.from_pretrained(
                "models/huggingface.co/microsoft/deberta-v3-base"
            )
            
            # Load model
            self.router_model = DebertaV2ForSequenceClassification.from_pretrained(
                "models/huggingface.co/microsoft/deberta-v3-base", 
                num_labels=2 
            )
            
            # Load trained weights
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location=self.device)
                self.router_model.load_state_dict(state_dict)
                logger.info(f"Router model loaded successfully: {model_path}")
            else:
                logger.warning(f"Model file not found: {model_path}, using pretrained weights")
            
            self.router_model.to(self.device)
            self.router_model.eval()
            
        except Exception as e:
            logger.error(f"Router model loading failed: {e}")
            raise e
    
    def _predict_router(self, queries: List[str]) -> List[Tuple[float, float]]:
        """
        Predict retrieval method using dual-output router model
        
        Args:
            queries: List of queries
            
        Returns:
            List[Tuple[float, float]]: Prediction probabilities for each query [(passage_prob, graph_prob), ...]
        """
        predictions = []
        
        with torch.no_grad():
            for query in tqdm(queries, desc="Router prediction"):
                # Encode query
                inputs = self.router_tokenizer(
                    query,
                    truncation=True,
                    padding='max_length',
                    max_length=256,
                    return_tensors='pt'
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Predict
                outputs = self.router_model(**inputs)
                logits = outputs.logits  # shape (1, 2)
                # Use independent sigmoid for each output
                passage_prob = torch.sigmoid(logits[0, 0]).item()  # Independent passage probability
                graph_prob = torch.sigmoid(logits[0, 1]).item()     # Independent graph probability
                predictions.append((passage_prob, graph_prob))
        
        return predictions
    
    def _get_retrieval_strategy(self, query: str, passage_prob: float, graph_prob: float) -> str:
        """
        Determine retrieval strategy based on dual-output router prediction
        Uses graph_prob - passage_prob > threshold as decision criterion
        
        Args:
            query: Query string
            passage_prob: Passage probability
            graph_prob: Graph probability
            
        Returns:
            str: Retrieval strategy ("graph" or "passage")
        """
        if (graph_prob - passage_prob) > self.router_threshold:
            return "graph"
        else:
            return "passage"
    
    def _update_performance_metrics(self, retrieval_time: float = 0.0, qa_time: float = 0.0, 
                                  retrieval_metrics: Dict = None, qa_metrics: Dict = None):
        """
        Update performance metrics
        
        Args:
            retrieval_time: Time spent on retrieval (from HippoRAG.all_retrieval_time)
            qa_time: Time spent on QA (from HippoRAG.all_retrieval_time)
            retrieval_metrics: Retrieval evaluation metrics (from HippoRAG)
            qa_metrics: QA evaluation metrics (from HippoRAG)
        """
        # Update timing metrics
        self.performance_metrics["total_retrieval_time"] = retrieval_time
        self.performance_metrics["total_qa_time"] = qa_time
        
        # Update retrieval metrics directly from HippoRAG
        if retrieval_metrics:
            self.performance_metrics["retrieval_recall_at_k"] = retrieval_metrics
        
        # Update QA metrics
        if qa_metrics:
            if "ExactMatch" in qa_metrics:
                self.performance_metrics["qa_em_scores"] = [qa_metrics["ExactMatch"]]
            if "F1" in qa_metrics:
                self.performance_metrics["qa_f1_scores"] = [qa_metrics["F1"]]
    
    def _update_detailed_stats(self, strategy: str, query_count: int = 1, 
                             retrieval_time: float = 0.0, qa_time: float = 0.0,
                             retrieval_metrics: Dict = None, qa_metrics: Dict = None):
        """
        Update detailed statistics for specific strategy (graph or passage)
        
        Args:
            strategy: Strategy name ("graph" or "passage")
            query_count: Number of queries processed
            retrieval_time: Time spent on retrieval
            qa_time: Time spent on QA
            retrieval_metrics: Retrieval evaluation metrics (batch-level aggregated dict, e.g., {"recall@1": 0.5, "recall@5": 0.8})
            qa_metrics: QA evaluation metrics (batch-level aggregated dict, e.g., {"ExactMatch": 0.6, "F1": 0.7})
        """
        if strategy not in self.detailed_stats:
            return
            
        # Update query count
        self.detailed_stats[strategy]["query_count"] += query_count
        
        # Update timing metrics
        self.detailed_stats[strategy]["retrieval_time"] += retrieval_time
        self.detailed_stats[strategy]["qa_time"] += qa_time
        
        # Update retrieval metrics
        if retrieval_metrics:
            for k, v in retrieval_metrics.items():
                self.detailed_stats[strategy]["retrieval_recall_at_k"][k] = v
        
        # Update QA metrics
        if qa_metrics:
            if "ExactMatch" in qa_metrics:
                self.detailed_stats[strategy]["qa_em_score"] = qa_metrics["ExactMatch"]
            if "F1" in qa_metrics:
                self.detailed_stats[strategy]["qa_f1_score"] = qa_metrics["F1"]
        
        # Calculate averages
        self._calculate_strategy_averages(strategy)
    
    def _calculate_strategy_averages(self, strategy: str):
        """Calculate average metrics for specific strategy"""
        if strategy not in self.detailed_stats:
            return
            
        stats = self.detailed_stats[strategy]
        query_count = stats["query_count"]
        
        if query_count > 0:
            # Calculate average times
            stats["avg_retrieval_time_per_query"] = stats["retrieval_time"] / query_count
            stats["avg_qa_time_per_query"] = stats["qa_time"] / query_count
            
            # QA scores are already aggregated for the batch, so we can use them directly
            if stats["qa_em_score"] is not None:
                stats["avg_em_score"] = stats["qa_em_score"]
            if stats["qa_f1_score"] is not None:
                stats["avg_f1_score"] = stats["qa_f1_score"]
    
    def _calculate_average_metrics(self) -> Dict[str, Any]:
        """
        Calculate average performance metrics
        
        Returns:
            Dict containing averaged metrics
        """
        avg_metrics = {}
        
        # Use retrieval metrics directly from HippoRAG
        if self.performance_metrics["retrieval_recall_at_k"]:
            avg_metrics.update(self.performance_metrics["retrieval_recall_at_k"])
        
        # Calculate average QA scores
        if self.performance_metrics["qa_em_scores"]:
            avg_metrics["avg_em_score"] = self.performance_metrics["qa_em_scores"][0]
        if self.performance_metrics["qa_f1_scores"]:
            avg_metrics["avg_f1_score"] = self.performance_metrics["qa_f1_scores"][0]
        
        # Calculate timing metrics
        total_queries = self.router_stats["total_queries"]
        if total_queries > 0:
            avg_metrics["avg_retrieval_time_per_query"] = self.performance_metrics["total_retrieval_time"] / total_queries
            avg_metrics["avg_qa_time_per_query"] = self.performance_metrics["total_qa_time"] / total_queries
        
        return avg_metrics

    def _compute_overall_metrics_from_detailed(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Compute overall QA (ExactMatch/F1) and retrieval metrics by weighting
        graph/passages stats with their query counts.
        Returns (combined_qa_metrics, combined_retrieval_metrics)
        """
        detailed = self.get_detailed_stats()
        router_stats = self.get_router_stats()
        g_n = router_stats.get('graph_queries', 0)
        p_n = router_stats.get('passage_queries', 0)

        def wavg(vg: float | None, vp: float | None) -> float | None:
            num = 0.0
            denom = 0
            if vg is not None:
                num += vg * g_n
                denom += g_n
            if vp is not None:
                num += vp * p_n
                denom += p_n
            return (num / denom) if denom > 0 else None

        # QA metrics
        graph_em = detailed.get('graph', {}).get('avg_em_score')
        graph_f1 = detailed.get('graph', {}).get('avg_f1_score')
        passage_em = detailed.get('passage', {}).get('avg_em_score')
        passage_f1 = detailed.get('passage', {}).get('avg_f1_score')

        overall_em = wavg(graph_em, passage_em)
        overall_f1 = wavg(graph_f1, passage_f1)

        combined_qa_metrics: Dict[str, float] = {}
        if overall_em is not None:
            combined_qa_metrics['ExactMatch'] = overall_em
        if overall_f1 is not None:
            combined_qa_metrics['F1'] = overall_f1

        # Retrieval metrics (avg_xxx keys)
        graph_avg_ret = detailed.get('graph', {}).get('avg_retrieval_metrics', {}) or {}
        passage_avg_ret = detailed.get('passage', {}).get('avg_retrieval_metrics', {}) or {}
        keys = set(graph_avg_ret.keys()) | set(passage_avg_ret.keys())
        combined_retrieval_metrics: Dict[str, float] = {}
        for k in keys:
            combined_val = wavg(graph_avg_ret.get(k), passage_avg_ret.get(k))
            if combined_val is not None:
                combined_retrieval_metrics[k] = combined_val

        return combined_qa_metrics, (combined_retrieval_metrics or None)
    
    def index(self, docs: List[str]):
        """
        Build index for documents (for both HippoRAG and StandardRAG)
        
        Args:
            docs: List of documents
        """
        logger.info("Starting document indexing...")
        
        if self.hipporag:
            logger.info("Building index for HippoRAG...")
            self.hipporag.index(docs)
            logger.info("HippoRAG indexing completed")
        
        if self.standard_rag:
            logger.info("Building index for StandardRAG...")
            self.standard_rag.index(docs)
            logger.info("StandardRAG indexing completed")
    
    def rag_qa(self, 
               queries: List[str], 
               gold_docs: List[List[str]] = None,
               gold_answers: List[List[str]] = None,
               return_router_info: bool = False) -> Union[Tuple, Tuple]:
        """
        Execute retrieval-augmented question answering with router-based strategy selection
        
        Args:
            queries: List of queries
            gold_docs: Gold standard documents
            gold_answers: Gold standard answers
            return_router_info: Whether to return router info
            
        Returns:
            QA results, include router info
        """
        logger.info(f"Starting RAG QA for {len(queries)} queries...")
        start_time = time.time()
        
        # Use router to predict retrieval strategies
        router_predictions = self._predict_router(queries)
        
        avg_passage_prob = sum([p_pass for p_pass, _ in router_predictions]) / len(router_predictions)
        avg_graph_prob = sum([p_graph for _, p_graph in router_predictions]) / len(router_predictions)
        logger.info(f"Average passage probability: {avg_passage_prob}, Average graph probability: {avg_graph_prob}")
        import numpy as np
        
        # Extract passage and graph probabilities separately
        passage_probs = [p_pass for p_pass, _ in router_predictions]
        graph_probs = [p_graph for _, p_graph in router_predictions]
        
        # Add statistics text
        stats_text = f'Total queries: {len(router_predictions)}\n'
        stats_text += f'Avg passage prob: {np.mean(passage_probs):.3f}\n'
        stats_text += f'Avg graph prob: {np.mean(graph_probs):.3f}\n'
        stats_text += f'Passage prob std: {np.std(passage_probs):.3f}\n'
        stats_text += f'Graph prob std: {np.std(graph_probs):.3f}'
        
        retrieval_strategies = [self._get_retrieval_strategy(q, p_pass, p_graph) for q, (p_pass, p_graph) in zip(queries, router_predictions)]
        
        # Update statistics
        self.router_stats["total_queries"] += len(queries)
        self.router_stats["graph_queries"] += sum(1 for s in retrieval_strategies if s == "graph")
        self.router_stats["passage_queries"] += sum(1 for s in retrieval_strategies if s == "passage")
        
        logger.info(f"Router prediction results: Graph={self.router_stats['graph_queries']}, Passage={self.router_stats['passage_queries']}")
        
        # Group queries by strategy
        graph_queries = []
        passage_queries = []
        graph_indices = []
        passage_indices = []
        
        for i, (query, strategy) in enumerate(zip(queries, retrieval_strategies)):
            if strategy == "graph":
                graph_indices.append(i)
            else:  # passage
                passage_indices.append(i)
        
        # Single-step rewriting: after routing, if selected strategy prob < threshold
        final_queries = []
        rewritten_flags = []
        for i, (query, strategy, (passage_prob, graph_prob)) in enumerate(zip(queries, retrieval_strategies, router_predictions)):
            if self.query_rewriter:
                final_query = self.query_rewriter.rewrite(
                    query, strategy=strategy, passage_prob=passage_prob, graph_prob=graph_prob
                )
                final_queries.append(final_query)
                is_rewritten = final_query != query
                rewritten_flags.append(is_rewritten)
                if is_rewritten:
                    self.router_stats["rewritten_queries"] += 1
            else:
                final_queries.append(query)
                rewritten_flags.append(False)
        
        # Update query lists with final rewritten queries
        graph_queries = [final_queries[i] for i in graph_indices]
        passage_queries = [final_queries[i] for i in passage_indices]
        
        # Initialize result arrays
        all_query_solutions = [None] * len(queries)
        all_answers = [None] * len(queries)
        all_metadata = [None] * len(queries)
        router_info = []
        
        # Process graph queries with HippoRAG (single batch call)
        if graph_queries and self.hipporag:
            logger.info(f"Processing {len(graph_queries)} graph queries with HippoRAG...")
            # Prepare gold data for graph queries
            graph_gold_docs = [gold_docs[i] for i in graph_indices] if gold_docs else None
            graph_gold_answers = [gold_answers[i] for i in graph_indices] if gold_answers else None
            
            # Record retrieval time before processing
            retrieval_time_before = self.hipporag.all_retrieval_time if hasattr(self.hipporag, 'all_retrieval_time') else 0.0
            
            # Record start time for QA
            graph_qa_start_time = time.time()
            
            # Single batch call to HippoRAG
            result = self.hipporag.rag_qa(graph_queries, graph_gold_docs, graph_gold_answers)
            logger.info(f"graphrag result: {result[3]}, {result[4]}")
            
            # Calculate actual retrieval time
            actual_retrieval_time = 0.0
            if hasattr(self.hipporag, 'all_retrieval_time'):
                actual_retrieval_time = self.hipporag.all_retrieval_time - retrieval_time_before
            
            # Calculate QA time (approximate: total time minus retrieval time)
            graph_qa_time = time.time() - graph_qa_start_time
            
            # Extract results
            if len(result) >= 3:
                query_solutions, answers, metadata = result[:3]
                graph_retrieval_metrics = result[3] if len(result) >= 4 else None
                
                # Store results in correct positions
                for idx, (qs, ans, meta) in enumerate(zip(query_solutions, answers, metadata)):
                    original_idx = graph_indices[idx]
                    all_query_solutions[original_idx] = qs
                    all_answers[original_idx] = ans
                    all_metadata[original_idx] = meta
                
                # Extract and store QA metrics from HippoRAG
                if len(result) == 5:
                    qa_eval_metrics = result[4]  # QA evaluation metrics
                    self._update_detailed_stats(
                        strategy="graph",
                        query_count=len(graph_queries),
                        retrieval_time=actual_retrieval_time,
                        qa_time=graph_qa_time,
                        retrieval_metrics=graph_retrieval_metrics,
                        qa_metrics=qa_eval_metrics
                    )
                    logger.info(f"Graph strategy QA metrics: {qa_eval_metrics}")
            
            logger.info(f"Graph queries completed: {len(graph_queries)} queries")
        
        # Process passage queries with StandardRAG (single batch call)
        if passage_queries and self.standard_rag:
            logger.info(f"Processing {len(passage_queries)} passage queries with StandardRAG...")
            # Prepare gold data for passage queries
            passage_gold_docs = [gold_docs[i] for i in passage_indices] if gold_docs else None
            passage_gold_answers = [gold_answers[i] for i in passage_indices] if gold_answers else None
            
            # Record retrieval time before processing
            retrieval_time_before = self.standard_rag.all_retrieval_time if hasattr(self.standard_rag, 'all_retrieval_time') else 0.0
            
            # Record start time for QA
            passage_qa_start_time = time.time()
            
            # Single batch call to StandardRAG
            result = self.standard_rag.rag_qa(passage_queries, passage_gold_docs, passage_gold_answers)
            logger.info(f"passage rag result: {result[3]}, {result[4]}")

            # Calculate actual retrieval time
            actual_retrieval_time = 0.0
            if hasattr(self.standard_rag, 'all_retrieval_time'):
                actual_retrieval_time = self.standard_rag.all_retrieval_time - retrieval_time_before
            
            # Calculate QA time (approximate: total time minus retrieval time)
            passage_qa_time = time.time() - passage_qa_start_time
            
            # Extract results
            if len(result) >= 3:
                query_solutions, answers, metadata = result[:3]
                passage_retrieval_metrics = result[3] if len(result) >= 4 else None
                
                # Store results in correct positions
                for idx, (qs, ans, meta) in enumerate(zip(query_solutions, answers, metadata)):
                    original_idx = passage_indices[idx]
                    all_query_solutions[original_idx] = qs
                    all_answers[original_idx] = ans
                    all_metadata[original_idx] = meta
                
                # Extract and store QA metrics from StandardRAG
                if len(result) == 5:
                    qa_eval_metrics = result[4]  # QA evaluation metrics
                    self._update_detailed_stats(
                        strategy="passage",
                        query_count=len(passage_queries),
                        retrieval_time=actual_retrieval_time,
                        qa_time=passage_qa_time,
                        retrieval_metrics=passage_retrieval_metrics,
                        qa_metrics=qa_eval_metrics
                    )
                    logger.info(f"Passage strategy QA metrics: {qa_eval_metrics}")

            logger.info(f"Passage queries completed: {len(passage_queries)} queries")
        
        # Create router info for all queries
        for i, (original_query, (passage_prob, graph_prob), strategy) in enumerate(zip(queries, router_predictions, retrieval_strategies)):
            router_info.append({
                "query": original_query,
                "final_query": final_queries[i],
                "rewritten": rewritten_flags[i],
                "passage_prob": passage_prob,
                "graph_prob": graph_prob,
                "strategy": strategy,
                "threshold": self.router_threshold
            })
        
        # Log performance summary
        total_time = time.time() - start_time
        logger.info(f"QA completed in {total_time:.2f}s")

        # Compute retrieval/QA time deltas from underlying modules (best-effort)
        total_retrieval_time = 0.0
        if self.hipporag and hasattr(self.hipporag, 'all_retrieval_time'):
            total_retrieval_time += self.hipporag.all_retrieval_time
        if self.standard_rag and hasattr(self.standard_rag, 'all_retrieval_time'):
            total_retrieval_time += self.standard_rag.all_retrieval_time

        # Update overall performance metrics so summary has EM/F1 and times
        combined_qa_metrics, combined_retrieval_metrics = self._compute_overall_metrics_from_detailed()

        self._update_performance_metrics(
            retrieval_time=total_retrieval_time,
            qa_time=total_time,
            retrieval_metrics=combined_retrieval_metrics,
            qa_metrics=combined_qa_metrics if combined_qa_metrics else None
        )
        
        if return_router_info:
            return all_query_solutions, all_answers, all_metadata, router_info
        else:
            return all_query_solutions, all_answers, all_metadata
    
    def get_router_stats(self) -> Dict[str, Any]:
        """Get router statistics"""
        if self.router_stats["total_queries"] > 0:
            self.router_stats["graph_ratio"] = self.router_stats["graph_queries"] / self.router_stats["total_queries"]
            self.router_stats["passage_ratio"] = self.router_stats["passage_queries"] / self.router_stats["total_queries"]
            self.router_stats["rewrite_ratio"] = self.router_stats["rewritten_queries"] / self.router_stats["total_queries"]
        else:
            self.router_stats["rewrite_ratio"] = 0.0
        return self.router_stats.copy()
    
    def get_detailed_stats(self) -> Dict[str, Any]:
        """Get detailed statistics for graph and passage strategies"""
        # Calculate ratios for each strategy
        total_queries = self.router_stats["total_queries"]
        detailed_stats_copy = {}
        
        for strategy in ["graph", "passage"]:
            stats = self.detailed_stats[strategy].copy()
            
            # Add ratio information
            if total_queries > 0:
                stats["query_ratio"] = stats["query_count"] / total_queries
            else:
                stats["query_ratio"] = 0.0
            
            # Retrieval metrics are already aggregated for the batch, so we can use them directly
            if stats["retrieval_recall_at_k"]:
                avg_recall = {}
                for k, v in stats["retrieval_recall_at_k"].items():
                    avg_recall[f"avg_{k}"] = v
                stats["avg_retrieval_metrics"] = avg_recall
            
            detailed_stats_copy[strategy] = stats
        
        return detailed_stats_copy
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        avg_metrics = self._calculate_average_metrics()
        detailed_stats = self.get_detailed_stats()
        
        # Combine router stats and performance metrics
        combined_metrics = {
            "router_stats": self.router_stats.copy(),
            "performance_metrics": self.performance_metrics.copy(),
            "average_metrics": avg_metrics,
            "detailed_stats": detailed_stats
        }
        
        return combined_metrics
    
    def log_performance_summary(self):
        """Log comprehensive performance summary"""
        metrics = self.get_performance_metrics()
        
        logger.info("=" * 60)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("=" * 60)
        
        # Router statistics
        router_stats = metrics["router_stats"]
        logger.info(f"Router Statistics:")
        logger.info(f"  Total queries: {router_stats['total_queries']}")
        logger.info(f"  Graph queries: {router_stats['graph_queries']} ({router_stats.get('graph_ratio', 0):.2%})")
        logger.info(f"  Passage queries: {router_stats['passage_queries']} ({router_stats.get('passage_ratio', 0):.2%})")
        logger.info(f"  Rewritten queries: {router_stats.get('rewritten_queries', 0)} ({router_stats.get('rewrite_ratio', 0):.2%})")
        
        # Detailed statistics for each strategy
        detailed_stats = metrics["detailed_stats"]
        logger.info(f"\nDetailed Strategy Statistics:")
        logger.info("-" * 40)
        
        for strategy in ["graph", "passage"]:
            stats = detailed_stats[strategy]
            logger.info(f"\n{strategy.upper()} Strategy:")
            logger.info(f"  Query count: {stats['query_count']} ({stats['query_ratio']:.2%})")
            logger.info(f"  Total retrieval time: {stats['retrieval_time']:.2f}s")
            logger.info(f"  Total QA time: {stats['qa_time']:.2f}s")
            logger.info(f"  Avg retrieval time per query: {stats['avg_retrieval_time_per_query']:.4f}s")
            logger.info(f"  Avg QA time per query: {stats['avg_qa_time_per_query']:.4f}s")
            
            # QA performance metrics
            if stats.get('qa_em_score') is not None:
                logger.info(f"  Avg EM score: {stats['avg_em_score']:.4f}")
            if stats.get('qa_f1_score') is not None:
                logger.info(f"  Avg F1 score: {stats['avg_f1_score']:.4f}")
            
            # Retrieval performance metrics
            if stats.get('avg_retrieval_metrics'):
                logger.info(f"  Retrieval metrics:")
                for metric, value in stats['avg_retrieval_metrics'].items():
                    logger.info(f"    {metric}: {value:.4f}")
        
        # Overall timing metrics
        perf_metrics = metrics["performance_metrics"]
        logger.info(f"\nOverall Timing Metrics:")
        logger.info(f"  Total retrieval time: {perf_metrics['total_retrieval_time']:.2f}s")
        logger.info(f"  Total QA time: {perf_metrics['total_qa_time']:.2f}s")
        
        # Average metrics
        avg_metrics = metrics["average_metrics"]
        if avg_metrics:
            logger.info(f"\nOverall Performance Metrics:")
            for key, value in avg_metrics.items():
                if isinstance(value, float):
                    logger.info(f"  {key}: {value:.4f}")
                else:
                    logger.info(f"  {key}: {value}")
        
        logger.info("=" * 60)
    
    def reset_stats(self):
        """Reset all statistics"""
        self.router_stats = {
            "total_queries": 0,
            "graph_queries": 0,
            "passage_queries": 0,
            "rewritten_queries": 0,
            "router_accuracy": 0.0
        }
        
        self.performance_metrics = {
            "total_retrieval_time": 0.0,
            "total_qa_time": 0.0,
            "retrieval_recall_at_k": {},
            "qa_em_scores": [],
            "qa_f1_scores": []
        }
        
        # Reset detailed statistics
        self.detailed_stats = {
            "graph": {
                "query_count": 0,
                "retrieval_time": 0.0,
                "qa_time": 0.0,
                "retrieval_recall_at_k": {},
                "qa_em_score": None,
                "qa_f1_score": None,
                "avg_retrieval_time_per_query": 0.0,
                "avg_qa_time_per_query": 0.0,
                "avg_em_score": 0.0,
                "avg_f1_score": 0.0
            },
            "passage": {
                "query_count": 0,
                "retrieval_time": 0.0,
                "qa_time": 0.0,
                "retrieval_recall_at_k": {},
                "qa_em_score": None,
                "qa_f1_score": None,
                "avg_retrieval_time_per_query": 0.0,
                "avg_qa_time_per_query": 0.0,
                "avg_em_score": 0.0,
                "avg_f1_score": 0.0
            }
        }
        
        logger.info("All statistics have been reset")


def load_router_threshold(threshold_file: str) -> float:
    """
    Load best threshold from file
    
    Args:
        threshold_file: Path to threshold file
        
    Returns:
        float: Best threshold
    """
    try:
        if os.path.exists(threshold_file):
            with open(threshold_file, 'r') as f:
                threshold_info = json.load(f)
            return threshold_info.get("best_threshold", 0.5)
        else:
            logger.warning(f"Threshold file not found: {threshold_file}, using default threshold 0.5")
            return 0.5
    except Exception as e:
        logger.error(f"Failed to load threshold: {e}, using default threshold 0.5")
        return 0.5


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Router-integrated HippoRAG")
    parser.add_argument("--router_model_path", type=str, required=True,
                       help="Path to trained router model")
    parser.add_argument("--threshold_file", type=str, default=None,
                       help="Path to threshold file (containing best threshold)")
    parser.add_argument("--low_probability_threshold", type=float, default=0.7,
                       help="rewriter threshold")
    parser.add_argument("--dataset", type=str, default="musique",
                       help="Dataset name")
    parser.add_argument("--save_dir", type=str, default="outputs",
                       help="Output directory")
    parser.add_argument("--log_file", type=str, default=None,
                       help="Log file path for output")
    parser.add_argument("--llm_base_url", type=str, default="https://api.openai.com/v1",
                       help="LLM base URL")
    parser.add_argument("--llm_name", type=str, default="gpt-4o-mini",
                       help="LLM name")
    parser.add_argument("--embedding_name", type=str, default="nvidia/NV-Embed-v2",
                       help="Embedding model name")
    parser.add_argument("--force_rebuild", action="store_true",
                       help="Force rebuild index and OpenIE from scratch (default: use cache if available)")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Determine threshold
    if args.threshold_file:
        threshold = load_router_threshold(args.threshold_file)
    else:
        threshold = 0.0
    
    logger.info(f"Using router threshold: {threshold}")
    save_dir = args.save_dir + '/' + args.dataset

    # Load data
    corpus_path = f"HippoRAG/reproduce/dataset/{args.dataset}_corpus.json"
    with open(corpus_path, "r") as f:
        corpus = json.load(f)
    
    docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]
    
    # Create configuration
    config = BaseConfig(
        save_dir=save_dir,
        llm_base_url=args.llm_base_url,
        llm_name=args.llm_name,
        dataset=args.dataset,
        embedding_model_name=args.embedding_name,
        force_index_from_scratch=args.force_rebuild,
        force_openie_from_scratch=args.force_rebuild,
        rerank_dspy_file_path="HippoRAG/src/hipporag/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json",
        retrieval_top_k=200,
        linking_top_k=5,
        max_qa_steps=3,
        qa_top_k=5,
        graph_type="facts_and_sim_passage_node_unidirectional",
        embedding_batch_size=8,
        max_new_tokens=None,
        corpus_len=len(corpus),
        openie_mode="online"
    )
    
    # Initialize Router-integrated HippoRAG
    router_hipporag = RouterIntegratedHippoRAG(
        router_model_path=args.router_model_path,
        router_threshold=threshold,
        hipporag_config=config,
        low_probability_threshold=args.low_probability_threshold,
        log_file=args.log_file
    )
    
    # Build index
    logger.info("Starting indexing...")
    router_hipporag.index(docs)
    logger.info("Indexing completed")
    
    # Load test queries
    samples_path = f"HippoRAG/reproduce/dataset/{args.dataset}.json"
    with open(samples_path, "r") as f:
        samples = json.load(f)
    
    test_queries = [s['question'] for s in samples]
    logger.info(f"Processing {len(test_queries)} test queries")
    
    # Get gold documents and answers for evaluation
    gold_docs = get_gold_docs(samples, args.dataset)
    gold_answers = get_gold_answers(samples)
    logger.info(f"Loaded {len(gold_docs)} gold document sets and {len(gold_answers)} gold answer sets")
    
    # Execute QA
    logger.info("Starting QA...")
    qa_results = router_hipporag.rag_qa(test_queries, gold_docs=gold_docs, gold_answers=gold_answers, return_router_info=True)
    
    # Save results to file
    if len(qa_results) >= 4:
        query_solutions, answers, metadata, router_info = qa_results
        logger.info("QA completed, saving results to file...")
        result_file = args.log_file.replace('.log', '_results.json')
        result_data = []
        for i, (query, query_solution, response, info) in enumerate(zip(test_queries, query_solutions, answers, router_info)):
            result_data.append({
                "query": query,
                "strategy": info['strategy'],
                "passage_prob": info['passage_prob'],
                "graph_prob": info['graph_prob'],
                "rewritten": info['rewritten'],
                "rewritten_query": info['final_query'],
                "response": response,
                "predicted_answer": query_solution.answer,
                "gold_answer": list(gold_answers[i])
            })
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to: {result_file}")
    
    # Log comprehensive performance summary
    router_hipporag.log_performance_summary()
    
    # Get detailed metrics
    metrics = router_hipporag.get_performance_metrics()
    logger.info("Detailed performance metrics saved to log file")
    
    # Save metrics to file if log file is specified
    if args.log_file:
        metrics_file = args.log_file.replace('.log', '_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Performance metrics saved to: {metrics_file}")


if __name__ == "__main__":
    main()
