# -*- coding: utf-8 -*-
"""
NV-Embed Embedding Model implementation for GraphRAG
Custom embedding model wrapper for NVIDIA NV-Embed (local only)
"""

import logging
from typing import Any, List, Union
import asyncio
from concurrent.futures import ThreadPoolExecutor
import torch
import sys
import os
import numpy as np


from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.language_model.protocol.base import EmbeddingModel

logger = logging.getLogger(__name__)


class NVEmbedEmbeddingModel(EmbeddingModel):
    """
    NV-Embed Embedding Model implementation for GraphRAG (local only)
    
    This class wraps NVIDIA NV-Embed model to work with GraphRAG.
    Only supports local model inference via HuggingFace transformers.
    """
    
    def __init__(
        self,
        config: LanguageModelConfig,
        model_name: str = "nvidia/NV-Embed-v2",
        device: str = "cuda",
        **kwargs: Any
    ):
        """
        Initialize NV-Embed Embedding Model
        
        Args:
            config: LanguageModelConfig instance
            model_name: NV-Embed model name or local path (default: "nvidia/NV-Embed-v2")
            device: Device to use for local inference ("cuda" or "cpu")
            **kwargs: Additional arguments
        """
        self.config = config
        self.model_name = model_name
        self.device = device
        
        # Initialize the embedding model
        self._model = None
        self._tokenizer = None
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize the model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize NV-Embed model using HuggingFace transformers"""
        try:
            self._init_huggingface()
        except Exception as e:
            logger.error(f"Failed to initialize NV-Embed model: {e}")
            raise e
    
    def _init_huggingface(self):
        """Initialize NV-Embed via HuggingFace transformers"""
        try:
            from transformers import AutoModel
            
            if torch is None:
                raise ImportError("torch is required for HuggingFace model loading")
            
            logger.info(f"Loading NV-Embed model from local: {self.model_name}")
            # NV-Embed uses AutoModel and has an encode method
            self._model = AutoModel.from_pretrained(
                self.model_name,
                device_map="auto" if torch.cuda.is_available() and self.device == "cuda" else "cpu",
                torch_dtype=torch.float16 if torch.cuda.is_available() and self.device == "cuda" else torch.float32,
                trust_remote_code=True
            )
            self._model.eval()
            logger.info(f"Initialized NV-Embed via HuggingFace: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize NV-Embed via HuggingFace: {e}")
            raise e
    
    def _encode_batch(self, texts: List[str], batch_size: int = 16, max_length: int = 32768, instruction: str = "") -> np.ndarray:
        """
        Encode a batch of texts using NV-Embed's encode method.
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for processing
            max_length: Maximum sequence length
            instruction: Optional instruction prefix
            
        Returns:
            numpy array of embeddings [batch_size, hidden_size]
        """
        if not hasattr(self._model, 'encode'):
            raise ValueError("NV-Embed model does not have encode method")
        
        # Prepare encode parameters (following NVEmbedV2.py pattern)
        encode_params = {
            "max_length": max_length,
            "instruction": instruction,
        }
        
        # Process in batches if needed
        if len(texts) <= batch_size:
            encode_params["prompts"] = texts
            results = self._model.encode(**encode_params)
        else:
            # Process in chunks
            results = []
            num_chunks = (len(texts) + batch_size - 1) // batch_size
            for i in range(0, len(texts), batch_size):
                chunk = texts[i:i + batch_size]
                chunk_idx = i // batch_size + 1
                encode_params["prompts"] = chunk
                chunk_results = self._model.encode(**encode_params)
                results.append(chunk_results)
            
            # Concatenate results
            if isinstance(results[0], torch.Tensor):
                results = torch.cat(results, dim=0)
            elif isinstance(results[0], np.ndarray):
                results = np.concatenate(results, axis=0)
            else:
                # Fallback: convert to tensor then concatenate
                results = [torch.tensor(r) if not isinstance(r, torch.Tensor) else r for r in results]
                results = torch.cat(results, dim=0)
        
        # Convert to numpy if needed
        if isinstance(results, torch.Tensor):
            results = results.cpu().numpy()
        
        return results
    
    def _embed_with_local(self, text: Union[str, List[str]], **kwargs: Any) -> Union[List[float], List[List[float]]]:
        """
        Generate embedding using local model (supports single text or batch)
        
        Uses NV-Embed's encode method following the pattern from NVEmbedV2.py
        """
        is_batch = isinstance(text, list)
        texts = text if is_batch else [text]
        
        # Get parameters from kwargs or use defaults
        batch_size = kwargs.get("batch_size", 16)
        max_length = kwargs.get("max_length", 32768)
        instruction = kwargs.get("instruction", "")
        
        try:
            # Use NV-Embed's encode method
            embeddings = self._encode_batch(texts, batch_size=batch_size, max_length=max_length, instruction=instruction)
            
            # Check if embeddings is valid
            if embeddings is None or embeddings.size == 0:
                logger.error(f"[NV-Embed] Encoding returned empty result: embeddings={embeddings}")
                raise ValueError("Encoding returned empty result")
            
            # Convert to list format
            if embeddings.ndim == 1:
                # Single embedding
                result = embeddings.tolist()
                return result if not is_batch else [result]
            else:
                # Batch embeddings: [batch_size, hidden_size]
                result = embeddings.tolist()
                return result if is_batch else result[0]
                
        except Exception as e:
            logger.error(f"[NV-Embed] Error in _embed_with_local: {e}, text type: {type(text)}, is_batch: {is_batch}, text sample: {str(text)[:100] if text else 'None'}")
            import traceback
            logger.error(f"[NV-Embed] Traceback: {traceback.format_exc()}")
            raise
    
    def embed(self, text: str, **kwargs: Any) -> List[float]:
        """
        Generate an embedding vector for the given text (synchronous)
        
        Args:
            text: The text to generate an embedding for
            **kwargs: Additional keyword arguments
            
        Returns:
            A list of floats representing the embedding vector
        """
        result = self._embed_with_local(text, **kwargs)
        return result
    
    async def aembed(self, text: str, **kwargs: Any) -> List[float]:
        """
        Generate an embedding vector for the given text (asynchronous)
        
        Args:
            text: The text to generate an embedding for
            **kwargs: Additional keyword arguments
            
        Returns:
            A list of floats representing the embedding vector
        """
        # Run local embedding in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._embed_with_local,
            text
        )
    
    def embed_batch(self, text_list: List[str], **kwargs: Any) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts (synchronous)
        
        Args:
            text_list: List of texts to generate embeddings for
            **kwargs: Additional keyword arguments (batch_size, max_length, instruction)
            
        Returns:
            List of embedding vectors
        """
        if not text_list:
            logger.warning("[NV-Embed] embed_batch() called with empty text_list")
            return []
        
        # Use batch processing with encode method
        result = self._embed_with_local(text_list, **kwargs)
        
        # Ensure result is a list of lists
        if result and isinstance(result[0], (int, float)):
            # Single embedding returned, wrap it
            return [result]
        
        return result
    
    async def aembed_batch(self, text_list: List[str], **kwargs: Any) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts (asynchronous)
        
        Args:
            text_list: List of texts to generate embeddings for
            **kwargs: Additional keyword arguments (batch_size, max_length, instruction)
            
        Returns:
            List of embedding vectors
        """
        if not text_list:
            return []
        
        # Process batch in thread pool (batch processing is more efficient)
        # Use lambda to pass kwargs
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._embed_with_local(text_list, **kwargs)
        )
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)

