# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""HuggingFace Tokenizer for local models."""

from typing import Optional

from graphrag.tokenizer.tokenizer import Tokenizer


class HuggingFaceTokenizer(Tokenizer):
    """HuggingFace Tokenizer for local models that use their own tokenizer."""

    def __init__(self, model_name: str) -> None:
        """Initialize the HuggingFace Tokenizer.

        Args
        ----
            model_name (str): The name or path of the HuggingFace model to use for tokenization.
        """
        self.model_name = model_name
        self._tokenizer = None
        self._initialize_tokenizer()

    def _initialize_tokenizer(self) -> None:
        """Initialize the HuggingFace tokenizer."""
        try:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        except Exception as e:
            raise ImportError(
                f"Failed to load HuggingFace tokenizer for model {self.model_name}: {e}. "
                "Please ensure transformers is installed and the model path is correct."
            ) from e

    def encode(self, text: str) -> list[int]:
        """Encode the given text into a list of tokens.

        Args
        ----
            text (str): The input text to encode.

        Returns
        -------
            list[int]: A list of tokens representing the encoded text.
        """
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")
        # Use the tokenizer to encode text, return token IDs
        return self._tokenizer.encode(text, add_special_tokens=False)

    def decode(self, tokens: list[int]) -> str:
        """Decode a list of tokens back into a string.

        Args
        ----
            tokens (list[int]): A list of tokens to decode.

        Returns
        -------
            str: The decoded string from the list of tokens.
        """
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")
        return self._tokenizer.decode(tokens, skip_special_tokens=True)

