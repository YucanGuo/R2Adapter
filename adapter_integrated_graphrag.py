# -*- coding: utf-8 -*-
"""
Router-integrated GraphRAG code file
Decides query retrieval method based on trained router model:
- If router predicts graph (>threshold), use GraphRAG for graph retrieval
- If router predicts passage (<=threshold), use BasicSearch for dense retrieval
"""

import os
import json
import torch
import numpy as np
import pandas as pd
import asyncio
from typing import List, Union, Tuple, Dict, Any, Optional
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
import logging
from tqdm import tqdm
from datetime import datetime
from rewriter import QueryRewriter
import time
import sys
import re
import string
from collections import Counter

# Add graphrag folder to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
graphrag_dir = os.path.join(current_dir, 'graphrag')
sys.path.append('graphrag')

# Import GraphRAG modules
from graphrag.api.index import build_index
from graphrag.api.query import (
    global_search,
    local_search,
    basic_search,
)
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.config.load_config import load_config
from graphrag.config.enums import IndexingMethod
from graphrag.storage.file_pipeline_storage import FilePipelineStorage


sys.path.append('HippoRAG/src')
from hipporag.StandardRAG import StandardRAG
from hipporag.utils.config_utils import BaseConfig
from hipporag.llm import _get_llm_class
from hipporag.prompts.prompt_template_manager import PromptTemplateManager

logger = logging.getLogger(__name__)


def _ensure_openai_api_key(base_url: str):
    """
    Ensure OPENAI_API_KEY is set when calling OpenAI-compatible endpoints.
    For local vLLM (http://localhost:8000/v1), set a dummy key if missing.
    """
    if base_url and "localhost" in base_url and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "sk-"  # dummy key for local endpoints
        logger.info("OPENAI_API_KEY was missing; set dummy key for local endpoint.")


def normalize_answer(answer: str) -> str:
    """
    Normalize a given string by applying the following transformations:
    1. Convert the string to lowercase.
    2. Remove punctuation characters.
    3. Remove the articles "a", "an", and "the".
    4. Normalize whitespace by collapsing multiple spaces into one.
    
    Reference: MRQA official eval / HippoRAG eval_utils
    """
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    
    def white_space_fix(text):
        return " ".join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(answer))))


def compute_exact_match(prediction: str, gold_answers: List[str]) -> float:
    """
    Compute exact match score
    
    Args:
        prediction: Predicted answer string
        gold_answers: List of gold answer strings
        
    Returns:
        float: 1.0 if prediction exactly matches any gold answer, 0.0 otherwise
    """
    if not prediction or not gold_answers:
        return 0.0
    
    normalized_pred = normalize_answer(prediction)
    # Check against all gold answers and take the maximum (best match)
    em_scores = [1.0 if normalize_answer(str(gold)) == normalized_pred else 0.0 for gold in gold_answers]
    return float(np.max(em_scores)) if em_scores else 0.0


def compute_f1(prediction: str, gold_answers: List[str]) -> float:
    """
    Compute F1 score based on token overlap (considering token frequency)
    
    Args:
        prediction: Predicted answer string
        gold_answers: List of gold answer strings
        
    Returns:
        float: F1 score (0.0 to 1.0), taking the maximum across all gold answers
    """
    if not prediction or not gold_answers:
        return 0.0
    
    def compute_f1_single(gold: str, predicted: str) -> float:
        """Compute F1 score between a single gold answer and predicted answer"""
        gold_tokens = normalize_answer(gold).split()
        predicted_tokens = normalize_answer(predicted).split()
        
        if not gold_tokens or not predicted_tokens:
            return 0.0
        
        # Use Counter to consider token frequency
        common = Counter(predicted_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            return 0.0
        
        precision = 1.0 * num_same / len(predicted_tokens)
        recall = 1.0 * num_same / len(gold_tokens)
        return 2 * (precision * recall) / (precision + recall)
    
    # Compute F1 for each gold answer and take the maximum
    f1_scores = [compute_f1_single(str(gold), prediction) for gold in gold_answers]
    return float(np.max(f1_scores)) if f1_scores else 0.0


def compute_qa_metrics(predictions: List[str], gold_answers: List[List[str]]) -> Dict[str, float]:
    """
    Compute QA metrics (EM and F1) for a batch of predictions
    
    Args:
        predictions: List of predicted answer strings
        gold_answers: List of gold answer lists (each element is a list of possible answers)
        
    Returns:
        Dict[str, float]: Dictionary containing 'ExactMatch' and 'F1' scores (averaged across examples)
    """
    if not predictions or not gold_answers:
        return {"ExactMatch": 0.0, "F1": 0.0}
    
    if len(predictions) != len(gold_answers):
        logger.warning(f"Mismatch in predictions ({len(predictions)}) and gold_answers ({len(gold_answers)}) length")
        return {"ExactMatch": 0.0, "F1": 0.0}
    
    total_em = 0.0
    total_f1 = 0.0
    
    for pred, gold_list in zip(predictions, gold_answers):
        # Convert gold_list to list of strings if needed
        if isinstance(gold_list, set):
            gold_list = list(gold_list)
        elif not isinstance(gold_list, list):
            gold_list = [str(gold_list)]
        else:
            gold_list = [str(g) for g in gold_list]
        
        # Compute EM and F1 (already returns float, taking max across gold answers)
        em = compute_exact_match(pred, gold_list)
        f1 = compute_f1(pred, gold_list)
        
        total_em += em
        total_f1 += f1
    
    avg_em = total_em / len(predictions) if predictions else 0.0
    avg_f1 = total_f1 / len(predictions) if predictions else 0.0
    
    return {"ExactMatch": avg_em, "F1": avg_f1}


def postprocess_graphrag_response(response: str, query: str, qa_llm=None, prompt_template_manager=None) -> str:
    """
    Post-process GraphRAG response using HippoRAG's QA prompt to ensure consistent format.
    Reference: "To ensure a consistent evaluation, the same QA prompt that HippoRAG 2 adopts 
    from HippoRAG (Gutiérrez et al., 2024) is applied to rephrase the original response of GraphRAG"
    
    Args:
        response: Raw response from GraphRAG (treated as retrieved context)
        query: Original query
        qa_llm: LLM instance for rephrasing
        prompt_template_manager: HippoRAG prompt template manager
        
    Returns:
        str: Rephrased answer following HippoRAG format, or original response if LLM unavailable
    """
    if not response or not query:
        return response if response else ""
    
    # If LLM and prompt manager are not available, return original response
    if qa_llm is None or prompt_template_manager is None:
        logger.info("QA LLM not available, returning original GraphRAG response")
        return extract_short_answer(response)
    
    # Use GraphRAG response as context (similar to retrieved passages in HippoRAG)
    # Format: "Wikipedia Title: <response_content>"
    prompt_user = f'Wikipedia Title: {response}\n\n'
    prompt_user += f'Question: {query}\nThought: '
    
    # Use MUSIQUE prompt template
    dataset_name = 'musique'  # Default template
    qa_messages = prompt_template_manager.render(
        name=f'rag_qa_{dataset_name}', 
        prompt_user=prompt_user
    )
    # Call LLM to rephrase the answer
    response_content, metadata, cache_hit = qa_llm.infer(qa_messages)
    # Extract answer
    return extract_short_answer(response_content)
        

def extract_short_answer(text: str) -> str:
    """
    Extract a concise answer from a model response.
    Prioritizes content after 'Answer:' if present; otherwise uses the first line.
    """
    if not text:
        return ""
    cleaned = text.strip()
    if "Answer:" in cleaned:
        cleaned = cleaned.split("Answer:", 1)[1].strip()
    return cleaned


def get_gold_docs(samples: List, dataset_name: str = None) -> List:
    """Extract gold documents from samples for evaluation"""
    gold_docs = []
    for sample in samples:
        if 'supporting_facts' in sample:  # hotpotqa, 2wikimultihopqa
            gold_title = set([item[0] for item in sample['supporting_facts']])
            gold_title_and_content_list = [item for item in sample['context'] if item[0] in gold_title]
            if dataset_name and dataset_name.startswith('hotpotqa'):
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


class RouterIntegratedGraphRAG:
    """
    Router-integrated GraphRAG class
    Selects retrieval strategy based on router model predictions
    """
    
    def __init__(self, 
                 router_model_path: str,
                 router_threshold: float = 0.5,
                 graphrag_config: GraphRagConfig = None,
                 standardrag_config: Optional[BaseConfig] = None,
                 enable_query_rewriting: bool = True,
                 low_probability_threshold: float = 0.6,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 log_file: str = None,
                 graph_search_type: str = "local",  # "local", "global", or "drift"
                 community_level: int = 0):
        """
        Initialize Router-integrated GraphRAG
        
        Args:
            router_model_path: Path to trained router model
            router_threshold: Router prediction threshold, above which uses GraphRAG, otherwise BasicSearch
            graphrag_config: GraphRAG configuration
            standardrag_config: BaseConfig for DPR passage retrieval (optional)
            enable_query_rewriting: Whether to enable query rewriting
            low_probability_threshold: Low probability threshold for query rewriting
            device: Computing device
            log_file: Log file path for output
            graph_search_type: Type of graph search ("local", "global", or "drift")
            community_level: Community level for graph search
        """
        self.device = device
        self.router_threshold = 0.0  # router_threshold
        self.log_file = log_file
        self.graphrag_config = graphrag_config
        self.graph_search_type = graph_search_type
        self.community_level = community_level
        self.standardrag_config = standardrag_config
        
        # Setup logging
        self._setup_logging()
        
        # Load router model
        self._load_router_model(router_model_path)
        
        if enable_query_rewriting:
            # Get LLM config from GraphRAG config
            model_config = graphrag_config.get_language_model_config(
                graphrag_config.local_search.chat_model_id
            )
            if hasattr(model_config, 'api_base') and model_config.api_base:
                llm_base_url = model_config.api_base
            if hasattr(model_config, 'model') and model_config.model:
                llm_name = model_config.model
            logger.info(f"Query rewriter configured: base_url={llm_base_url}, model={llm_name}")
            
            self.query_rewriter = QueryRewriter(
                llm_base_url=llm_base_url,
                llm_name=llm_name,
                low_probability_threshold=low_probability_threshold
            )
        else:
            self.query_rewriter = None
        
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
                "qa_em_scores": [],
                "qa_f1_scores": [],
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
                "qa_em_scores": [],
                "qa_f1_scores": [],
                "avg_retrieval_time_per_query": 0.0,
                "avg_qa_time_per_query": 0.0,
                "avg_em_score": 0.0,
                "avg_f1_score": 0.0
            }
        }
        
        # GraphRAG data storage
        self.graphrag_data = {
            "entities": None,
            "communities": None,
            "community_reports": None,
            "text_units": None,
            "relationships": None,
            "covariates": None
        }
        
        # Initialize LLM and prompt template manager for post-processing GraphRAG responses
        # This ensures consistent output format with HippoRAG
        self.qa_llm = None
        self.prompt_template_manager = None
        # Get LLM config from GraphRAG config
        model_config = graphrag_config.get_language_model_config(
            graphrag_config.local_search.chat_model_id
        )
        llm_base_url = model_config.api_base
        llm_name = model_config.model
        _ensure_openai_api_key(llm_base_url)
        # Get embedding model config from GraphRAG config
        embedding_model_config = graphrag_config.get_language_model_config(
            graphrag_config.local_search.embedding_model_id
        )
        embedding_model_name = embedding_model_config.model_name

        # DPR retriever for passage routing
        standardrag_config.llm_base_url = llm_base_url
        standardrag_config.llm_name = llm_name
        standardrag_config.embedding_model_name = embedding_model_name
        self.standard_rag = StandardRAG(global_config=standardrag_config)
        logger.info("Initialized StandardRAG for passage/DPR retrieval.")

        # Create a minimal BaseConfig for LLM initialization
        if BaseConfig:
            qa_config = BaseConfig()
            qa_config.llm_base_url = llm_base_url
            qa_config.llm_name = llm_name
            qa_config.save_dir = graphrag_config.output.base_dir if hasattr(graphrag_config.output, 'base_dir') else "./outputs"
            
            self.qa_llm = _get_llm_class(qa_config)
            self.prompt_template_manager = PromptTemplateManager()
            logger.info(f"Initialized QA LLM for post-processing: base_url={llm_base_url}, model={llm_name}")
           
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
    
    
    def _load_graphrag_data(self, output_dir: str):
        """Load GraphRAG index data from parquet files"""
        try:
            storage = FilePipelineStorage(base_dir=output_dir)
            
            # Load entities
            entities_path = os.path.join(output_dir, "entities.parquet")
            if os.path.exists(entities_path):
                self.graphrag_data["entities"] = pd.read_parquet(entities_path)
                logger.info(f"Loaded entities: {len(self.graphrag_data['entities'])} entities")
            
            # Load communities
            communities_path = os.path.join(output_dir, "communities.parquet")
            if os.path.exists(communities_path):
                self.graphrag_data["communities"] = pd.read_parquet(communities_path)
                logger.info(f"Loaded communities: {len(self.graphrag_data['communities'])} communities")
            
            # Load community reports
            community_reports_path = os.path.join(output_dir, "community_reports.parquet")
            if os.path.exists(community_reports_path):
                self.graphrag_data["community_reports"] = pd.read_parquet(community_reports_path)
                logger.info(f"Loaded community reports: {len(self.graphrag_data['community_reports'])} reports")
            
            # Load text units
            text_units_path = os.path.join(output_dir, "text_units.parquet")
            if os.path.exists(text_units_path):
                self.graphrag_data["text_units"] = pd.read_parquet(text_units_path)
                logger.info(f"Loaded text units: {len(self.graphrag_data['text_units'])} text units")
            
            # Load relationships
            relationships_path = os.path.join(output_dir, "relationships.parquet")
            if os.path.exists(relationships_path):
                self.graphrag_data["relationships"] = pd.read_parquet(relationships_path)
                logger.info(f"Loaded relationships: {len(self.graphrag_data['relationships'])} relationships")
            
            # Load covariates (optional)
            covariates_path = os.path.join(output_dir, "covariates.parquet")
            if os.path.exists(covariates_path):
                self.graphrag_data["covariates"] = pd.read_parquet(covariates_path)
                logger.info(f"Loaded covariates: {len(self.graphrag_data['covariates'])} covariates")
            
        except Exception as e:
            logger.error(f"Failed to load GraphRAG data: {e}")
            raise e
    
    async def _index_async(self, docs: List[str] = None, output_dir: str = None, force_rebuild: bool = False):
        """
        Build index for documents using GraphRAG (async version)
        
        Args:
            docs: List of documents (optional if index already exists)
            output_dir: Output directory for GraphRAG index
            force_rebuild: Whether to force rebuild index
        """
        if not self.graphrag_config:
            raise ValueError("GraphRAG config is required for indexing")
        
        output_dir = output_dir or self.graphrag_config.output.base_dir
        
        # Check if index already exists; if so, reuse unless force_rebuild=True
        existing_index = os.path.exists(os.path.join(output_dir, "text_units.parquet"))
        if not force_rebuild and existing_index:
            logger.info("Index already exists, loading from disk without rebuilding...")
            self._load_graphrag_data(output_dir)
            # Still allow building DPR index if docs are provided
            if self.standard_rag and docs:
                logger.info("Building/refreshing DPR (StandardRAG) index from provided docs...")
                self.standard_rag.index(docs)
            return
        
        if docs is None:
            raise ValueError("Documents are required for building index when no cache is available or force_rebuild=True")
        
        logger.info("Starting GraphRAG indexing...")
        
        # Get current timestamp as ISO8601 string for creation_date
        current_date = datetime.now().isoformat()
        
        # Convert docs to DataFrame format expected by GraphRAG
        # According to GraphRAG docs, DataFrame must include: id, text, title, creation_date, metadata
        documents_df = pd.DataFrame({
            'id': [f"doc_{i}" for i in range(len(docs))],
            'text': docs,
            'title': [f"Document {i}" for i in range(len(docs))],
            'creation_date': [current_date] * len(docs),  # Use current timestamp for all documents
            'metadata': [json.dumps({})] * len(docs)  # Empty metadata as JSON string for each document
        })
        
        # Build index
        try:
            results = await build_index(
                config=self.graphrag_config,
                method=IndexingMethod.Standard,
                is_update_run=False,
                input_documents=documents_df,
                verbose=True
            )
            logger.info("GraphRAG indexing completed")
            
            # Load the created index data
            self._load_graphrag_data(output_dir)

            # Build DPR (StandardRAG) index if available
            if self.standard_rag:
                logger.info("Building DPR (StandardRAG) index for passage retrieval...")
                self.standard_rag.index(docs)
            
        except Exception as e:
            logger.error(f"GraphRAG indexing failed: {e}")
            raise e
    

    def index(self, docs: List[str] = None, output_dir: str = None, force_rebuild: bool = False):
        """
        Build index for documents using GraphRAG (synchronous wrapper)
        
        Args:
            docs: List of documents (optional if index already exists)
            output_dir: Output directory for GraphRAG index
            force_rebuild: Whether to force rebuild index
        """
        # Run async method
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If event loop is already running, try to use nest_asyncio
                try:
                    import nest_asyncio
                    nest_asyncio.apply()
                except ImportError:
                    logger.warning("nest_asyncio not available. If you're in an async context, consider using _index_async() instead.")
                    # Create a new event loop in a separate thread
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        def run_async():
                            return asyncio.run(
                                self._index_async(docs, output_dir, force_rebuild)
                            )
                        future = executor.submit(run_async)
                        return future.result()
        except RuntimeError:
            # No event loop exists, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self._index_async(docs, output_dir, force_rebuild)
        )
    
    async def _graph_search(self, query: str) -> Tuple[str, Dict]:
        """
        Perform graph search using GraphRAG
        
        Args:
            query: Query string
            
        Returns:
            Tuple[str, Dict]: (response, context_data)
        """
        if not self.graphrag_config:
            raise ValueError("GraphRAG config is required")
        
        if self.graph_search_type == "local":
            if self.graphrag_data["text_units"] is None or self.graphrag_data["entities"] is None:
                raise ValueError("GraphRAG data not loaded. Please run index() first.")
            
            if not hasattr(self.graphrag_config, 'vector_store') or not self.graphrag_config.vector_store:
                raise ValueError("Vector store configuration is missing!")
            
            try:
                response, context_data = await local_search(
                    config=self.graphrag_config,
                    entities=self.graphrag_data["entities"],
                    communities=self.graphrag_data["communities"],
                    community_reports=self.graphrag_data["community_reports"],
                    text_units=self.graphrag_data["text_units"],
                    relationships=self.graphrag_data["relationships"],
                    covariates=self.graphrag_data["covariates"],
                    community_level=self.community_level,
                    response_type="multiple paragraphs",
                    query=query,
                    verbose=False
                )
            except Exception as e:
                logger.error(f"Error in local_search: {e}")
                raise
        elif self.graph_search_type == "global":
            if self.graphrag_data["entities"] is None or self.graphrag_data["community_reports"] is None:
                raise ValueError("GraphRAG data not loaded. Please run index() first.")
            
            response, context_data = await global_search(
                config=self.graphrag_config,
                entities=self.graphrag_data["entities"],
                communities=self.graphrag_data["communities"],
                community_reports=self.graphrag_data["community_reports"],
                community_level=self.community_level,
                dynamic_community_selection=False,
                response_type="multiple paragraphs",
                query=query,
                verbose=False
            )
        else:
            raise ValueError(f"Unknown graph search type: {self.graph_search_type}")
        
        return response, context_data
    
    async def _basic_search(self, query: str) -> Tuple[str, Dict]:
        """
        Perform passage search using DPR (StandardRAG) if available; 
        otherwise fall back to GraphRAG BasicSearch
        
        Args:
            query: Query string
            
        Returns:
            Tuple[str, Dict]: (response, context_data)
        """
        # Prefer DPR (StandardRAG) for speed if configured
        if self.standard_rag:
            result = self.standard_rag.rag_qa([query])
            if len(result) >= 3:
                _, answers, metadata = result[:3]
                answer = answers[0] if answers else ""
                meta = metadata[0] if metadata else {}
                return answer, meta
            return "", {}
        
        # Fallback: GraphRAG BasicSearch
        if not self.graphrag_config:
            raise ValueError("GraphRAG config is required")
        
        if self.graphrag_data["text_units"] is None:
            raise ValueError("GraphRAG data not loaded. Please run index() first.")
        
        response, context_data = await basic_search(
            config=self.graphrag_config,
            text_units=self.graphrag_data["text_units"],
            query=query,
            verbose=False
        )
        
        return response, context_data
    
    async def _rag_qa_async(self, 
                            queries: List[str], 
                            gold_docs: List[List[str]] = None,
                            gold_answers: List[List[str]] = None,
                            return_router_info: bool = False) -> Union[Tuple, Tuple]:
        """
        Execute retrieval-augmented question answering with router-based strategy selection (async version)
        
        Args:
            queries: List of queries
            gold_docs: Gold standard documents
            gold_answers: Gold standard answers
            return_router_info: Whether to return router info
            
        Returns:
            QA results, may include router info
        """
        logger.info(f"Starting RAG QA for {len(queries)} queries...")
        start_time = time.time()
        
        # Use router to predict retrieval strategies
        router_predictions = self._predict_router(queries)
        
        avg_passage_prob = sum([p_pass for p_pass, _ in router_predictions]) / len(router_predictions)
        avg_graph_prob = sum([p_graph for _, p_graph in router_predictions]) / len(router_predictions)
        logger.info(f"Average passage probability: {avg_passage_prob}, Average graph probability: {avg_graph_prob}")
        
        retrieval_strategies = [self._get_retrieval_strategy(q, p_pass, p_graph) for q, (p_pass, p_graph) in zip(queries, router_predictions)]
        
        # Update statistics
        self.router_stats["total_queries"] += len(queries)
        self.router_stats["graph_queries"] += sum(1 for s in retrieval_strategies if s == "graph")
        self.router_stats["passage_queries"] += sum(1 for s in retrieval_strategies if s == "passage")
        
        logger.info(f"Router prediction results: Graph={self.router_stats['graph_queries']}, Passage={self.router_stats['passage_queries']}")
        
        # Group queries by strategy
        graph_indices = []
        passage_indices = []
        
        for i, strategy in enumerate(retrieval_strategies):
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
        
        # Initialize result arrays
        all_answers = [None] * len(queries)
        all_metadata = [None] * len(queries)
        router_info = []
        
        # Process graph queries with GraphRAG
        if graph_indices:
            logger.info(f"Processing {len(graph_indices)} graph queries with GraphRAG...")
            graph_queries = [final_queries[i] for i in graph_indices]
            
            retrieval_time_before = time.time()
            for idx, query in enumerate(graph_indices):
                original_idx = graph_indices[idx]
                query_text = graph_queries[idx]
                
                try:
                    response, context_data = await self._graph_search(query_text)
                    # Post-process using HippoRAG's QA prompt to ensure consistent format
                    response = postprocess_graphrag_response(
                        response, 
                        query_text, 
                        qa_llm=self.qa_llm,
                        prompt_template_manager=self.prompt_template_manager
                    )
                    all_answers[original_idx] = response
                    all_metadata[original_idx] = context_data
                    gold_answer = str(gold_answers[original_idx])
                    logger.info(f"gold answer: {gold_answer}")
                    logger.info(f"predicted answer: {all_answers[original_idx]}")
                except Exception as e:
                    logger.error(f"Error processing graph query {original_idx}: {e}")
                    all_answers[original_idx] = ""
                    all_metadata[original_idx] = {}
            
            actual_retrieval_time = time.time() - retrieval_time_before
            
            # Compute QA metrics for graph queries if gold_answers provided
            graph_qa_metrics = None
            if gold_answers:
                graph_answers = [all_answers[i] if all_answers[i] else "" for i in graph_indices]
                graph_gold_answers = [gold_answers[i] for i in graph_indices]
                graph_qa_metrics = compute_qa_metrics(graph_answers, graph_gold_answers)
                logger.info(f"Graph strategy QA metrics: EM={graph_qa_metrics['ExactMatch']:.4f}, F1={graph_qa_metrics['F1']:.4f}")
            
            self._update_detailed_stats(
                strategy="graph",
                query_count=len(graph_indices),
                retrieval_time=actual_retrieval_time,
                retrieval_metrics=None,  # GraphRAG doesn't provide retrieval metrics directly
                qa_metrics=graph_qa_metrics
            )
            
            logger.info(f"Graph queries completed: {len(graph_indices)} queries")
        
        # Process passage queries with BasicSearch
        if passage_indices:
            logger.info(f"Processing {len(passage_indices)} passage queries with BasicSearch...")
            passage_queries = [final_queries[i] for i in passage_indices]
            
            retrieval_time_before = time.time()
            for idx, query in enumerate(passage_indices):
                original_idx = passage_indices[idx]
                query_text = passage_queries[idx]
                
                try:
                    response, context_data = await self._basic_search(query_text)
                    response = extract_short_answer(response)
                    all_answers[original_idx] = response
                    all_metadata[original_idx] = context_data
                    gold_answer = str(gold_answers[original_idx])
                    logger.info(f"gold answer: {gold_answer}")
                    logger.info(f"predicted answer: {all_answers[original_idx]}")
                except Exception as e:
                    logger.error(f"Error processing passage query {original_idx}: {e}")
                    all_answers[original_idx] = ""
                    all_metadata[original_idx] = {}
            
            actual_retrieval_time = time.time() - retrieval_time_before
            
            # Compute QA metrics for passage queries if gold_answers provided
            passage_qa_metrics = None
            if gold_answers:
                passage_answers = [all_answers[i] if all_answers[i] else "" for i in passage_indices]
                passage_gold_answers = [gold_answers[i] for i in passage_indices]
                passage_qa_metrics = compute_qa_metrics(passage_answers, passage_gold_answers)
                logger.info(f"Passage strategy QA metrics: EM={passage_qa_metrics['ExactMatch']:.4f}, F1={passage_qa_metrics['F1']:.4f}")
            
            self._update_detailed_stats(
                strategy="passage",
                query_count=len(passage_indices),
                retrieval_time=actual_retrieval_time,
                retrieval_metrics=None,
                qa_metrics=passage_qa_metrics
            )
            
            logger.info(f"Passage queries completed: {len(passage_indices)} queries")
        
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
        
        # Compute overall QA metrics if gold_answers provided
        overall_qa_metrics = None
        if gold_answers:
            # Ensure all answers are strings
            all_answers_str = [ans if ans else "" for ans in all_answers]
            overall_qa_metrics = compute_qa_metrics(all_answers_str, gold_answers)
            logger.info(f"Overall QA metrics: EM={overall_qa_metrics['ExactMatch']:.4f}, F1={overall_qa_metrics['F1']:.4f}")
        
        # Update overall performance metrics
        total_retrieval_time = self.detailed_stats["graph"]["retrieval_time"] + self.detailed_stats["passage"]["retrieval_time"]
        
        self._update_performance_metrics(
            retrieval_time=total_retrieval_time,
            qa_time=total_time,
            retrieval_metrics=None,
            qa_metrics=overall_qa_metrics
        )
        
        if return_router_info:
            return all_answers, all_metadata, router_info
        else:
            return all_answers, all_metadata
    
    def rag_qa(self, 
               queries: List[str], 
               gold_docs: List[List[str]] = None,
               gold_answers: List[List[str]] = None,
               return_router_info: bool = False) -> Union[Tuple, Tuple]:
        """
        Execute retrieval-augmented question answering with router-based strategy selection
        (Synchronous wrapper for async method)
        
        Args:
            queries: List of queries
            gold_docs: Gold standard documents
            gold_answers: Gold standard answers
            return_router_info: Whether to return router info
            
        Returns:
            QA results, may include router info
        """
        # Run async method
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If event loop is already running, try to use nest_asyncio
                try:
                    import nest_asyncio
                    nest_asyncio.apply()
                except ImportError:
                    logger.warning("nest_asyncio not available. If you're in an async context, consider using rag_qa_async() instead.")
                    # Create a new event loop in a separate thread
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        def run_async():
                            return asyncio.run(
                                self._rag_qa_async(queries, gold_docs, gold_answers, return_router_info)
                            )
                        future = executor.submit(run_async)
                        return future.result()
        except RuntimeError:
            # No event loop exists, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self._rag_qa_async(queries, gold_docs, gold_answers, return_router_info)
        )
    
    def _update_performance_metrics(self, retrieval_time: float = 0.0, qa_time: float = 0.0, 
                                  retrieval_metrics: Dict = None, qa_metrics: Dict = None):
        """
        Update performance metrics
        
        Args:
            retrieval_time: Time spent on retrieval
            qa_time: Time spent on QA
            retrieval_metrics: Retrieval evaluation metrics
            qa_metrics: QA evaluation metrics
        """
        # Update timing metrics
        self.performance_metrics["total_retrieval_time"] = retrieval_time
        self.performance_metrics["total_qa_time"] = qa_time
        
        # Update retrieval metrics
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
            retrieval_metrics: Retrieval evaluation metrics
            qa_metrics: QA evaluation metrics
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
                if k not in self.detailed_stats[strategy]["retrieval_recall_at_k"]:
                    self.detailed_stats[strategy]["retrieval_recall_at_k"][k] = []
                self.detailed_stats[strategy]["retrieval_recall_at_k"][k].append(v)
        
        # Update QA metrics
        if qa_metrics:
            if "ExactMatch" in qa_metrics:
                self.detailed_stats[strategy]["qa_em_scores"].append(qa_metrics["ExactMatch"])
            if "F1" in qa_metrics:
                self.detailed_stats[strategy]["qa_f1_scores"].append(qa_metrics["F1"])
        
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
            
            # Calculate average QA scores
            if stats["qa_em_scores"]:
                stats["avg_em_score"] = sum(stats["qa_em_scores"]) / len(stats["qa_em_scores"])
            if stats["qa_f1_scores"]:
                stats["avg_f1_score"] = sum(stats["qa_f1_scores"]) / len(stats["qa_f1_scores"])
    
    def _calculate_average_metrics(self) -> Dict[str, Any]:
        """
        Calculate average performance metrics
        
        Returns:
            Dict containing averaged metrics
        """
        avg_metrics = {}
        
        # Use retrieval metrics directly
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
            
            # Calculate average retrieval metrics
            if stats["retrieval_recall_at_k"]:
                avg_recall = {}
                for k, values in stats["retrieval_recall_at_k"].items():
                    if values:
                        avg_recall[f"avg_{k}"] = sum(values) / len(values)
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
            if stats['qa_em_scores']:
                logger.info(f"  Avg EM score: {stats['avg_em_score']:.4f}")
            if stats['qa_f1_scores']:
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
                "qa_em_scores": [],
                "qa_f1_scores": [],
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
                "qa_em_scores": [],
                "qa_f1_scores": [],
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
    
    parser = argparse.ArgumentParser(description="Router-integrated GraphRAG")
    parser.add_argument("--router_model_path", type=str, required=True,
                       help="Path to trained router model")
    parser.add_argument("--threshold_file", type=str, default=None,
                       help="Path to threshold file (containing best threshold)")
    parser.add_argument("--low_probability_threshold", type=float, default=0.6,
                       help="Rewriter threshold")
    parser.add_argument("--dataset", type=str, default="musique",
                       help="Dataset name")
    parser.add_argument("--save_dir", type=str, default="outputs",
                       help="Output directory")
    parser.add_argument("--log_file", type=str, default=None,
                       help="Log file path for output")
    parser.add_argument("--config_file", type=str, required=True,
                       help="Path to GraphRAG config YAML file")
    parser.add_argument("--graph_search_type", type=str, default="local",
                       choices=["local", "global"],
                       help="Type of graph search to use")
    parser.add_argument("--community_level", type=int, default=0,
                       help="Community level for graph search")
    parser.add_argument("--force_rebuild", action="store_true",
                       help="Force rebuild index from scratch")
    
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

    # Try to register NV-Embed model if available
    try:
        from graphrag.register_nvembed_model import register_nvembed_model
        register_nvembed_model()
        logger.info("NV-Embed model registered successfully")
    except ImportError as e:
        logger.debug(f"NV-Embed model registration skipped: {e}")
    except Exception as e:
        logger.warning(f"Failed to register NV-Embed model: {e}")

    # Load GraphRAG config
    # load_config requires root_dir and config_filepath
    from pathlib import Path
    config_file_path = Path(args.config_file)
    # Use current working directory as root_dir, and pass config file path
    root_dir = Path.cwd()
    config = load_config(root_dir=root_dir, config_filepath=config_file_path)
    config.output.base_dir = save_dir
    
    # Set vector store URI to be relative to output directory
    # This ensures index and query use the same vector store path
    if hasattr(config, 'vector_store') and config.vector_store:
        for store_name, store_config in config.vector_store.items():
            if hasattr(store_config, 'type') and store_config.type == 'lancedb':
                save_dir_abs = os.path.abspath(save_dir)
                vector_store_path = save_dir_abs
                # Set db_uri (handle both 'uri' and 'db_uri' fields)
                if hasattr(store_config, 'uri'):
                    store_config.db_uri = vector_store_path
                else:
                    store_config.db_uri = vector_store_path
                # Ensure container_name is set (defaults to "default" if not specified)
                if not hasattr(store_config, 'container_name') or store_config.container_name is None:
                    store_config.container_name = "default"
                logger.info(f"Vector store path configured: {vector_store_path}")
    
    # Load data
    corpus_path = f"HippoRAG/reproduce/dataset/{args.dataset}_corpus.json"
    with open(corpus_path, "r") as f:
        corpus = json.load(f)
    
    docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]

    # Create configuration
    standardrag_config = BaseConfig(
        save_dir=save_dir,
        dataset=args.dataset,
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
    
    # Initialize Router-integrated GraphRAG
    router_graphrag = RouterIntegratedGraphRAG(
        router_model_path=args.router_model_path,
        router_threshold=threshold,
        low_probability_threshold=args.low_probability_threshold,
        graphrag_config=config,
        standardrag_config=standardrag_config,
        log_file=args.log_file,
        graph_search_type=args.graph_search_type,
        community_level=args.community_level
    )
    
    # Build index
    logger.info("Starting indexing...")
    router_graphrag.index(docs, force_rebuild=args.force_rebuild)
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
    qa_results = router_graphrag.rag_qa(test_queries, gold_docs=gold_docs, gold_answers=gold_answers, return_router_info=True)
    
    if len(qa_results) >= 3:
        answers, metadata, router_info = qa_results
        logger.info("QA completed, sample results:")
        for i, (query, answer, info) in enumerate(zip(test_queries, answers, router_info)):
            # Show rewriting information
            if info['rewritten']:
                logger.info(f"Query {i+1}: {query}")
                logger.info(f"Strategy: {info['strategy']} (passage_prob: {info['passage_prob']:.4f}, graph_prob: {info['graph_prob']:.4f})")
                logger.info(f"Query rewritten: '{info['query']}' -> '{info['final_query']}'")
                logger.info(f"Answer: {answer[:200]}...")  # Truncate long answers
                logger.info("---")
    
    # Log comprehensive performance summary
    router_graphrag.log_performance_summary()
    
    # Get detailed metrics
    metrics = router_graphrag.get_performance_metrics()
    logger.info("Detailed performance metrics saved to log file")
    
    # Save metrics to file if log file is specified
    if args.log_file:
        metrics_file = args.log_file.replace('.log', '_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Performance metrics saved to: {metrics_file}")


if __name__ == "__main__":
    main()