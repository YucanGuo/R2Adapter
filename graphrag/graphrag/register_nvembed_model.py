# -*- coding: utf-8 -*-
"""
Register NV-Embed model with GraphRAG ModelFactory
Run this script before using adapter_integrated_graphrag.py with NV-Embed
"""

import logging
import sys
import os

from graphrag.language_model.factory import ModelFactory
from graphrag.nvembed_embedding_model import NVEmbedEmbeddingModel

logger = logging.getLogger(__name__)


def register_nvembed_model():
    """
    Register NV-Embed embedding model with GraphRAG ModelFactory
    
    This allows you to use NV-Embed in GraphRAG configuration files by setting:
    type: nvembed_embedding
    
    The model will be created with config and additional parameters from the config file.
    """
    def create_nvembed_model(**kwargs):
        """Create NV-Embed model instance"""
        # Extract config from kwargs
        config = kwargs.get('config')
        if config is None:
            raise ValueError("config is required for NV-Embed model")
        
        # Extract model_name: first from kwargs, then from config object
        # (now that model_name is a defined field in LanguageModelConfig)
        model_name = kwargs.get('model_name') or config.model_name
        if not model_name:
            # Fallback to model field (but this is usually the HuggingFace model name, not local path)
            model_name = config.model
            logger.warning(
                f"model_name not found in config, using model field '{model_name}'. "
                "Please ensure model_name is set in config.yaml for local model path."
            )
        
        # Extract device: first from kwargs, then from config object
        # (now that device is a defined field in LanguageModelConfig)
        device = kwargs.get('device') or config.device or 'cuda'
        
        logger.info(f"Creating NV-Embed model with model_name={model_name}, device={device}")
        
        return NVEmbedEmbeddingModel(
            config=config,
            model_name=model_name,
            device=device
        )
    
    # Register NV-Embed as a custom embedding model type
    ModelFactory.register_embedding(
        "nvembed_embedding",
        create_nvembed_model
    )
    
    logger.info("NV-Embed embedding model registered successfully!")
    logger.info("You can now use 'type: nvembed_embedding' in your GraphRAG config")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    register_nvembed_model()

