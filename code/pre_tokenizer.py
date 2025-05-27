from functools import partial
from typing import List
from utils import run_data_job_in_parallel
from patterns_and_dicts import SPLITER_REGEX, BEGINNING_OF_WORD_MARK, HASHTAG_TOKEN, USER_TOKEN, URL_TOKEN
import re


def pre_tokenize_text_file(pre_tokenizer, batch_of_text: tuple) -> List[str]:
    """
    pre tokenize a text file in parallel.
    Args:
        batch_of_text: tuple of List of text strings, starting index to process).
        pre_tokenizer: PreTokenizer instance.
    Returns: The list of pre token lists.
    """
    text, start_idx = batch_of_text
    return [pre_tokenizer.pre_tokenize_str(text=t) for t in text]


class PreTokenizer:
    def __init__(self,
                 split_punctuation: bool = False,
                 custom_spliter: str = None,
                 train_mode: bool = False):
        """
        Initialize the Pre_Tokenizer.

        Args:
            split_whitespace (bool): If True, the tokenizer will split tokens based on whitespace characters.
            split_punctuation (bool): If True, the tokenizer will split tokens based on punctuation marks.
            custom_spliter (str): A custom regular expression string to use as the splitting pattern.
                                  If set to "PRE DEFINED", a predefined regex constant (SPLITER_REGEX) will be used.
                                  If None, no custom splitting pattern is applied.
            train_mode: Weather to append special whitespace and beginning of a word marks. Default is false.
        """
        self.split_punctuation = split_punctuation

        self.custom_spliter = custom_spliter if custom_spliter is not None \
            else SPLITER_REGEX if custom_spliter == "PRE DEFINED" else None

        self.train_mode = train_mode

    def pre_tokenize_batch(self, text:List[str]) -> List[List[str]]:
        """
        pre tokenize a batch of normalized text strings
        Args:
            text: List of normalized text strings and start index of the batch from the overall data
        Returns: The list of the normalized text strings
        """
        job = partial(pre_tokenize_text_file, self)
        return run_data_job_in_parallel(data=text, job=job)

    def pre_tokenize_str(self, text: str) -> List[str]:
        """
        Split a string of normalized text to a pre token list
        Args:
            text: string of normalized text
        Returns: List of pre tokens
        """
        if self.custom_spliter:
            return re.findall(self.custom_spliter, text)
        else:
            # Use regex to find words and spaces
            pre_tokens_output = text.split()
            if not self.train_mode:
                pre_tokens_output = [BEGINNING_OF_WORD_MARK + pre_token for pre_token in pre_tokens_output]

            if self.split_punctuation:
                after_punctuation_split = []
                for pre_token in pre_tokens_output:
                    # split pre token by punctuation retain the punctuation
                    # Build regex string safely (escaping special characters)
                    special_tokens = [re.escape(tok) for tok in [
                        BEGINNING_OF_WORD_MARK,
                        HASHTAG_TOKEN,
                        USER_TOKEN,
                        URL_TOKEN
                    ]]

                    # Join them into a non-capturing group
                    special_token_pattern = "|".join(special_tokens)

                    # Full pattern: match special tokens OR word characters OR punctuation
                    pattern = rf"(?:{special_token_pattern})|\w+|[^\w\s]"
                    pre_token_splitted_by_punctuation = re.findall(pattern, pre_token)
                    if not self.train_mode:
                        merged_splited_word_start_mark = []
                        i = 0
                        while i < len(pre_token_splitted_by_punctuation):
                            if pre_token_splitted_by_punctuation[i] == BEGINNING_OF_WORD_MARK:
                                to_merge = pre_token_splitted_by_punctuation[i] + pre_token_splitted_by_punctuation[i+1]
                                i += 2
                            else:
                                to_merge = pre_token_splitted_by_punctuation[i]
                                i += 1
                            merged_splited_word_start_mark.append(to_merge)
                        pre_token_splitted_by_punctuation = merged_splited_word_start_mark
                    after_punctuation_split.extend(pre_token_splitted_by_punctuation)

                pre_tokens_output = after_punctuation_split
            return pre_tokens_output

# text_file = []
# with open("../data/small_test_data.txt", 'r', encoding='utf-8') as f:
#     text_file = f.readlines()
#
# normalizer = Normalizer(unicode_normalization='NFKD',
#                         lower_case="TITLE CASE + STOP WORDS",
#                         remove_accents=True,
#                         expand_contractions=True,
#                         replace_urls=True,
#                         replace_usernames=True,
#                         replace_hashtag=True,
#                         replace_html_tags=True)
# text = normalize_text_file(normalizer, (text_file, 0))

#text = "[USER] i had that exp yesterday...movie and dinner afteer a long long time...and let me tell you...it is not that great!"
# pre_tokenizer = PreTokenizer(train_mode=False,
#                              split_punctuation=True)
# pre_tokenize_text = pre_tokenize_text_file(pre_tokenizer, (text, 0))
# print(len(pre_tokenize_text))
# print(pre_tokenize_text[15])
# print(type(pre_tokenize_text))
#
# for t in pre_tokenize_text:
#     print(t)
# print("--- before pre tokenization")
# print(text)
# print("--- after pre tokenization")
# print(pre_tokenizer.pre_tokenize_str(text))
