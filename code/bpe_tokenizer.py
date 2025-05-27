from abc import ABC
from typing import List
from base_tokenizer import BaseTokenizer
from normalizer import Normalizer, normalize_text_file
from pre_tokenizer import PreTokenizer, pre_tokenize_text_file


class BPETokenizer(BaseTokenizer, ABC):
    def __init__(self,
                 vocab_size: int = 5000,
                 normalizer: Normalizer = None,
                 pre_tokenizer: PreTokenizer = None):
        BaseTokenizer.__init__()
        pre_tokens = {}  # words characters frequencies mapping
        merge_cand = {}  # merge candidates intermediate tokens frequencies mapping
        rules = {}  # merged tokens frequencies mapping
        vocab = set()  # set of all tokens that were learned during training time
        self.vocab_size = vocab_size
        if normalizer is None:
            normalizer = Normalizer(unicode_normalization='NFKD',
                                    lower_case="TITLE CASE + STOP WORDS",
                                    remove_accents=True,
                                    expand_contractions=True,
                                    replace_urls=True,
                                    replace_usernames=True,
                                    replace_hashtag=True,
                                    replace_html_tags=True)
        if pre_tokenizer is None:
            pre_tokenizer = PreTokenizer(train_mode=True,
                                         split_punctuation=True)
        self.normalizer = normalizer
        self.pre_tokenizer = pre_tokenizer

    def train(self, texts: List[str]) -> None:
        """
        Train the tokenizer on a list of texts
        Args:
            texts: List of training texts
        """
        normalized_text_file = normalize_text_file(normalizer=self.normalizer,
                                                   batch_of_text=(texts, 0))
        self.pre_tokenizer.train_mode = True
        pre_tokens_lists = pre_tokenize_text_file(pre_tokenizer=self.pre_tokenizer,
                                                  batch_of_text=(normalized_text_file, 0))
        pass

    def encode(self, text: str) -> List[int]:
        """
        Convert a text string into a list of token IDs
        Args:
            text: The input text to encode

        Returns:
            A list of token IDs
        """
        normalized_text = self.normalizer.normalize_text(text=text)
        self.pre_tokenizer.train_mode = False
        pre_tokens_list = self.pre_tokenizer.pre_tokenize_str(text=normalized_text)
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

    def initial_spliter(self, text: str) -> List[str]:
        """
        Args:
            text:
        Returns:
        """
        pass

    def _get_stats(self, ids):
        for pair in zip(ids, ids[1:]):
            self.merge_cand[pair] = self.merge_cand.get(pair, 0) + 1
