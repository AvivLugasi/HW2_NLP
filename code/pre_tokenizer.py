from functools import partial
from typing import List
from utils import run_data_job_in_parallel, logging
from patterns_and_dicts import SPLITER_REGEX, BEGINNING_OF_WORD_MARK, HASHTAG_TOKEN, USER_TOKEN, URL_TOKEN
import re


def pre_tokenize_text_file(pre_tokenizer, batch_of_text: tuple) -> List[List[str]]:
    """
    Pre‐tokenize a text file in parallel.

    Args:
        pre_tokenizer: PreTokenizer instance.
        batch_of_text: (List[str], start_idx) – a batch of normalized strings.
    Returns:
        A list of lists of pre‐tokens for each input string.
    """
    logging.info(f"==== Starting pre tokenization of text (total lines = {len(batch_of_text)}) ====\n")
    logging.info(f"==== pre tokenization flags ====\n"
                 f"\tsplit_punctuation={pre_tokenizer.split_punctuation}\n"
                 f"\tcustom_spliter={pre_tokenizer.custom_spliter}\n"
                 f"\ttrain_mode={pre_tokenizer.train_mode}\n")
    text, start_idx = batch_of_text
    pre_tokenized_text = [pre_tokenizer.pre_tokenize_str(text=t) for t in text]
    logging.info(f"==== Finished pre tokenization ====\n")
    return pre_tokenized_text


class PreTokenizer:
    def __init__(self,
                 split_punctuation: bool = False,
                 custom_spliter: str = None,
                 train_mode: bool = False):
        """
        Initialize the PreTokenizer.

        Args:
            split_punctuation (bool): If True, split tokens by punctuation (while still preserving
                                       special markers like "<W>", "[USER]", etc.).
            custom_spliter (str): If set to "PRE DEFINED", use SPLITER_REGEX; if another regex,
                                  use that. If None, split on whitespace only.
            train_mode (bool): Controls whether to use multi‐threaded batching (True) or not (False),
                               but DOES NOT affect whether we prefix "<W>": we always will.
        """
        self.split_punctuation = split_punctuation
        if custom_spliter is not None:
            if custom_spliter == "PRE DEFINED":
                self.custom_spliter = SPLITER_REGEX
            else:
                self.custom_spliter = custom_spliter
        else:
            self.custom_spliter = None

        # NOTE: train_mode no longer controls "<W>"‐prefixing. We ALWAYS prefix "<W>".
        self.train_mode = train_mode

    def pre_tokenize_batch(self, text: List[str]) -> List[List[str]]:
        """
        Pre‐tokenize a batch of normalized text strings in parallel.
        """
        job = partial(pre_tokenize_text_file, self)
        return run_data_job_in_parallel(data=text, job=job)

    def pre_tokenize_str(self, text: str) -> List[str]:
        """
        Split a normalized string into pre‐tokens.  Every original "word" (split on whitespace)
        will be prefixed with "<W>".  If split_punctuation=True, we further split by punctuation,
        but the "<W>" stays attached to the first sub‐token of each original word.

        Returns: List of pre‐tokens (with exactly one "<W>" at each “word” boundary).
        """
        # 1) If custom_spliter is provided, just use it:
        if self.custom_spliter:
            return re.findall(self.custom_spliter, text)

        # 2) Otherwise, split on whitespace first:
        raw_tokens = text.split()  # e.g. ["hello-world!", "I’m", "fine."]
        #  Always prefix each raw token with "<W>"
        prefixed_tokens = [BEGINNING_OF_WORD_MARK + tok for tok in raw_tokens]
        # e.g. ["<W>hello-world!", "<W>I’m", "<W>fine."]

        if not self.split_punctuation:
            # No punctuation splitting: just return as‐is
            return prefixed_tokens

        # 3) If split_punctuation=True, we must split each prefixed_token on punctuation,
        #    but we want to keep "<W>" and attach it to the first resulting piece.
        all_splits: List[str] = []
        # Build a regex that recognizes these “special tokens” as atomic:
        special_tokens = [
            re.escape(BEGINNING_OF_WORD_MARK),
            re.escape(HASHTAG_TOKEN),
            re.escape(USER_TOKEN),
            re.escape(URL_TOKEN)
        ]
        special_token_pattern = r"(?:%s)" % "|".join(special_tokens)
        # Final pattern: either <W> (or other special tokens), or alphanumeric, or single punctuation.
        # This ensures we never break "<W>" away from the first sub‐token, because "<W>" is matched as one ///
        # atomic token.
        pattern = rf"{special_token_pattern}|\w+|[^\w\s]"

        for pref in prefixed_tokens:
            # Example: pref = "<W>hello-world!"
            subtoks = re.findall(pattern, pref)
            # e.g. subtoks might be ["<W>", "hello", "-", "world", "!"]
            # Now we must merge "<W>" with the very next piece ("hello"), so it becomes "<W>hello":
            merged: List[str] = []
            i = 0
            while i < len(subtoks):
                tok = subtoks[i]
                if tok == BEGINNING_OF_WORD_MARK:
                    # Merge this with subtoks[i+1] (safe because every pref was "<W>..."):
                    if i + 1 < len(subtoks):
                        merged_tok = BEGINNING_OF_WORD_MARK + subtoks[i + 1]
                        merged.append(merged_tok)
                        i += 2
                    else:
                        # Edge case: "<W>" was the only piece (rare if the word were literally empty);
                        # keep it alone.
                        merged.append(BEGINNING_OF_WORD_MARK)
                        i += 1
                else:
                    # Just a normal sub‐token (either a punctuation or a standalone word piece)
                    merged.append(tok)
                    i += 1
            # e.g. merged might be ["<W>hello", "-", "world", "!"]
            all_splits.extend(merged)

        return all_splits


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

# text = "[USER] i had that exp yesterday...movie and dinner afteer a long long time...and let me tell you...it is not that great!"
# pre_tokenizer = PreTokenizer(train_mode=False,
#                              split_punctuation=True)
#
# tokenized_text = pre_tokenizer.pre_tokenize_str(text=text)
# print(tokenized_text)

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
