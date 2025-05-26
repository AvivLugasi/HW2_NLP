from abc import ABC
from typing import List
from base_tokenizer import BaseTokenizer


class BPETokenizer(BaseTokenizer, ABC):
    def __init__(self):
        BaseTokenizer.__init__()
        pre_tokens = {}  # words characters frequencies mapping
        merge_cand = {}  # merge candidates intermediate tokens frequencies mapping
        rules = {}  # merged tokens frequencies mapping
        vocab = set()  # set of all tokens that were learned during training time

    def train(self, texts: List[str]) -> None:
        """
        Train the tokenizer on a list of texts

        Args:
            texts: List of training texts
        """
        pass

    def encode(self, text: str) -> List[int]:
        """
        Convert a text string into a list of token IDs

        Args:
            text: The input text to encode

        Returns:
            A list of token IDs
        """
        pass

    def decode(self, token_ids: List[int]) -> str:
        """
        Convert a list of token IDs back to a text string

        Args:
            token_ids: List of token IDs to decode

        Returns:
            The decoded text string
        """
        pass

    def get_stats(self, ids):
        for pair in zip(ids, ids[1:]):
            self.merge_cand[pair] = self.merge_cand.get(pair, 0) + 1
