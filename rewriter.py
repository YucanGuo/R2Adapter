# -*- coding: utf-8 -*-
"""
Query Rewriter for Router-integrated hybrid RAG
Rewrites queries after router determines retrieval method when the selected method is graph and its probability is below threshold
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
import time
from openai import OpenAI

logger = logging.getLogger(__name__)


class BaseRewriter(ABC):
    """Base class for query rewriters"""
    
    def __init__(self, llm_base_url: str = "https://api.openai.com/v1", 
                 llm_name: str = "gpt-4o-mini", 
                 api_key: str = None):
        """
        Initialize rewriter
        
        Args:
            llm_base_url: LLM API base URL
            llm_name: LLM model name
            api_key: API key (optional; default fallback provided)
        """
        self.llm_base_url = llm_base_url
        self.llm_name = llm_name
        # Provide a default API key so that vLLM (OpenAI-compatible) works without credentials
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or "dummy-key"
        self._openai_client = OpenAI(
                base_url=self.llm_base_url,
                api_key=self.api_key
        )
    
    @abstractmethod
    def rewrite(self, query: str, **kwargs) -> str:
        """
        Rewrite query
        
        Args:
            query: Original query
            **kwargs: Additional parameters
            
        Returns:
            str: Rewritten query
        """
        pass
    
    def _ensure_openai_client(self):
        """Lazily initialize OpenAI-compatible client (works for OpenAI and vLLM)."""
        if self._openai_client is not None:
            return
        
        try:
            from openai import OpenAI
            self._openai_client = OpenAI(
                base_url=self.llm_base_url,
                api_key=self.api_key
            )
        except ImportError:
            raise RuntimeError("OpenAI client is not installed. Please install: pip install openai")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise e

    def _call_llm(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
        """
        Call LLM API
        
        Args:
            messages: List of messages
            temperature: Temperature parameter
            
        Returns:
            str: LLM response
        """
        try:
            response = self._openai_client.chat.completions.create(
                model=self.llm_name,
                messages=messages,
                temperature=temperature,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            return ""


class QueryRewriter(BaseRewriter):
    """
    Rewrites queries after router determines retrieval method when the selected method's probability is below threshold
    """
    
    def __init__(self, llm_base_url: str = "https://api.openai.com/v1", 
                 llm_name: str = "gpt-4o-mini", 
                 api_key: str = None,
                 low_probability_threshold: float = 0.7):
        """
        Initialize query rewriter
        
        Args:
            llm_base_url: LLM API base URL
            llm_name: LLM model name
            api_key: API key (optional)
            low_probability_threshold: Low probability threshold for triggering rewrite
        """
        super().__init__(llm_base_url, llm_name, api_key)
        self.low_probability_threshold = low_probability_threshold
    
    def should_rewrite(self, strategy: str, passage_prob: float, graph_prob: float) -> bool:
        """
        Determine if query needs rewriting based on selected strategy and its probability
        
        Args:
            strategy: Selected retrieval strategy ("graph" or "passage")
            passage_prob: Passage retrieval probability
            graph_prob: Graph retrieval probability
            
        Returns:
            bool: Whether rewriting is needed
        """
        if strategy == "graph":
            return graph_prob < self.low_probability_threshold
        elif strategy == "passage":
            return False
    
    def rewrite(self, query: str, strategy: str, passage_prob: float = None, graph_prob: float = None, **kwargs) -> str:
        """
        Rewrite query based on retrieval strategy and its probability
        
        Args:
            query: Original query
            strategy: Selected retrieval strategy ("graph" or "passage")
            passage_prob: Passage retrieval probability
            graph_prob: Graph retrieval probability
            **kwargs: Additional parameters
            
        Returns:
            str: Rewritten query
        """
        # Check if rewriting is needed
        if not self.should_rewrite(strategy, passage_prob or 0.0, graph_prob or 0.0):
            return query
        
        # logger.info(f"Starting query rewriting: {query} (graph solving probability: {graph_prob})")
        return self._rewrite_for_graph(query)
    
    def _rewrite_for_graph(self, query: str) -> str:
        """
        Rewrite query for knowledge graph retrieval
        
        Args:
            query: Original query
            
        Returns:
            str: Rewritten query
        """
        rewrite_prompt = f"""
You are an expert in query rewriting for **knowledge graph-based retrieval systems**.
Your task is to convert the given natural language question into one or more **structured triplets** of the form:

    [head_entity, relation, tail_entity]

Each triplet represents a relational fact or reasoning step used to answer the question.

### Rules
1. The final answer should be represented by **a triplet containing a question mark (?)** in place of the unknown target entity.
   - Example: "Who founded Tesla?" → `[?, founded, Tesla_Inc.]`
2. Not every triplet must contain a question mark. 
   - Intermediate steps may include unknown or inferred entities.
3. For **unknown intermediate entities**, use **angle brackets `<...>`** with a short descriptive phrase.
   - Example: `"Who taught the author of 'The Republic'?"` →  
     ```
     [<the author of 'The Republic'>, wrote, 'The Republic']
     [?, taught, <the author of 'The Republic'>]
     ```
4. The **order of triplets should follow the reasoning flow** required to answer the question (from known → inferred → target).
5. Avoid natural language explanations or commentary, only output the final structured reasoning chain.

### Output Format
Output as a **list of reasoning steps**, each in the following format:
    [head_entity, relation, tail_entity]

If multiple steps exist, write them in logical order.

Original question: {query}

Triplet reasoning chain:
"""
        
        messages = [
            {"role": "system", "content": "You are a professional query rewriting expert, specializing in optimizing queries for knowledge graph retrieval."},
            {"role": "user", "content": rewrite_prompt}
        ]
        
        rewritten_query = self._call_llm(messages)
        return query+'\nstructured triplet reasoning chain: '+rewritten_query.strip()