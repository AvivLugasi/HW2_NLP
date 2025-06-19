import itertools
import re
import string
from abc import ABC
from typing import List, Tuple, Dict, Set

from base_tokenizer import BaseTokenizer
from normalizer import Normalizer, normalize_text_file
from pre_tokenizer import PreTokenizer, pre_tokenize_text_file
from collections import Counter

from utils import logging

from patterns_and_dicts import SPECIAL_TOKENS, DETERMINERS, GPT4_SPLIT_PATTERN


class BPETokenizer(BaseTokenizer, ABC):
    def __init__(self,
                 vocab_size: int = 10000,
                 normalizer: Normalizer = None,
                 pre_tokenizer: PreTokenizer = None,
                 enable_bigrams: bool = True,
                 bigrams_freq_threshold: int = 10,
                 merge_freq_threshold: int = 20):
        """
        A Byte-Pair Encoding tokenizer that extends BaseTokenizer.
        Uses "<W>" as the word-begin marker to stay in sync with PreTokenizer.
        """
        # Initialize BaseTokenizer (sets up [PAD],[UNK],[BOS],[EOS])
        super().__init__()

        # BPE-specific containers
        self.corpus_dict: Dict[tuple[str], int] = {}
        self.merge_cand: Dict[Tuple[str, str], tuple[int, set]] = {}
        self.rules: List[Tuple[Tuple[str, str], str]] = []
        self.vocab: Set[str] = set()

        self.merge_ranks: Dict[Tuple[str, str], int] = {}

        # How many merge-iterations to run
        self.vocab_size = vocab_size

        # Marker for "beginning of word" (must match PreTokenizer's marker)
        self.space_token = "<W>"

        # append special tokens
        self._append_special_tokens()

        # Normalizer
        if normalizer is None:
            normalizer = Normalizer(
                unicode_normalization=None,
                lower_case="TITLE CASE + STOP WORDS",
                remove_accents=False,
                expand_contractions=False,
                replace_urls=False,
                replace_usernames=False,
                replace_hashtag=False,
                replace_html_tags=False,
                remove_repeated_letters=True,
                remove_suffix_and_prefix=False
            )
        self.normalizer = normalizer

        # PreTokenizer
        if pre_tokenizer is None:
            pre_tokenizer = PreTokenizer(
                train_mode=True,
                split_punctuation=True
            )
        self.pre_tokenizer = pre_tokenizer

        self.enable_bigrams = enable_bigrams
        self.bigrams_freq_threshold = bigrams_freq_threshold
        self.merge_freq_threshold = merge_freq_threshold

    def train(self, texts: List[str]) -> None:
        """
        BPE training via “Counter re‐count each iteration” to avoid scanning huge maps.
        Steps:
          1. Normalize & pre-tokenize the entire corpus.
          2. Build `tokenized_corpus`: a List[List[str]] of symbols (including "<W>").
          3. Repeat up to self.vocab_size merges:
             a. Count all adjacent pairs in the corpus with Counter.
             b. Pick the most common pair → merged_symbol.
             c. Rebuild every word that contains that pair, merging all occurrences.
          4. After merges, collect the final vocab and assign IDs.
        """
        logging.info(
            f"==== Starting incremental BPE training for {len(texts)} lines, up to {self.vocab_size} merges ====\n")

        # 1) Normalize all texts
        normalized_texts = normalize_text_file(
            normalizer=self.normalizer,
            batch_of_text=(texts, 0)
        )
        logging.info("==== Done Normalizing ====\n")

        # 2) Pre-tokenize in train_mode
        self.pre_tokenizer.train_mode = True
        pre_tokenized_sentences = pre_tokenize_text_file(
            pre_tokenizer=self.pre_tokenizer,
            batch_of_text=(normalized_texts, 0)
        )
        logging.info("==== Done Pre-tokenize ====\n")
        if self.enable_bigrams:
            pre_tokenized_sentences = self._create_bi_grams_pre_tokens(pre_tokenized_sentences)

        # 3) Build tokenized_corpus and possible merge dicts
        all_words = [w for line_tokens in pre_tokenized_sentences for w in line_tokens]

        self._build_corpus_dict(all_words=all_words)
        logging.info(f"==== Built tokenized_corpus (total words = {len(self.corpus_dict)}) ====\n")
        self._build_merge_cand_dict()
        merges_done = 0
        # 4) Perform merges up to vocab_size
        while len(self.vocab) < self.vocab_size:
            # 4a) return the pair with the highest
            best_pair, freq = self._get_best_pair()

            if best_pair is None or freq < self.merge_freq_threshold:
                print(best_pair)
                print(freq)
                break
            # define the merge
            merged_symbol = ''.join(best_pair)
            self.rules.append((best_pair, merged_symbol))
            # 4b) update the corpus and merge dicts
            self._update_corpus(best_pair)
            # 4c) remove the pair from the merge cand list
            del self.merge_cand[best_pair]
            # 4d) update rule and vocab
            self.vocab.add(merged_symbol)
            merges_done += 1
            logging.info(f"--- Merge #{merges_done}: {best_pair} → '{merged_symbol}' (freq={freq})")

        logging.info("==== Caching merge ranks for faster encoding... ====")
        self.merge_ranks = {pair: i for i, (pair, _) in enumerate(self.rules)}

        # 6) Assign token IDs to each subword (preserving [PAD],[UNK],[BOS],[EOS])
        next_id = max(self.token_to_id.values()) + 1
        for subword in sorted(self.vocab):
            if subword not in self.token_to_id:
                self.token_to_id[subword] = next_id
                self.id_to_token[next_id] = subword
                next_id += 1

        logging.info(
            f"==== Finished BPE training: {merges_done} merges done, final vocab size = {len(self.vocab) + 4} (including special tokens) ====\n")

    def encode(self, text: str) -> List[int]:
        """
        Convert a raw text string → list of token IDs via BPE.
        """
        normalized = self.normalizer.normalize_text(text=text)
        self.pre_tokenizer.train_mode = False
        pre_tokens = self.pre_tokenizer.pre_tokenize_str(text=normalized)

        all_subwords = []
        for word in pre_tokens:
            symbols = self._split_to_chars(word)
            merged_symbols = self._apply_merges(symbols)
            all_subwords.extend(merged_symbols)

        logging.debug(f"generated tokens: {all_subwords}")
        return [self.token_to_id.get(token, self.token_to_id["[UNK]"]) for token in all_subwords]

    def _apply_merges(self, symbols: List[str]) -> List[str]:
        """
        Repeatedly merges the highest-priority pair in the symbol list until no more merges are possible.
        """
        while len(symbols) > 1:
            # Find the next best merge operation.
            # Scans the list of symbols and finds the merge with the lowest rank (highest priority).
            min_rank = float('inf')
            best_pair_info = None  # Stores (index_of_merge, pair_to_merge)

            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                # Use the pre-computed cache for a fast O(1) lookup.
                rank = self.merge_ranks.get(pair)
                if rank is not None:
                    if rank < min_rank:
                        min_rank = rank
                        best_pair_info = (i, pair)

            # If no mergeable pair was found in the entire sequence, we are done.
            if best_pair_info is None:
                break

            # A merge was found, so we perform it.
            i, pair_to_merge = best_pair_info
            merged_symbol = ''.join(pair_to_merge)
            symbols = symbols[:i] + [merged_symbol] + symbols[i + 2:]

        return symbols

    def decode(self, token_ids: List[int]) -> str:
        """
        Convert a list of token IDs back to a text string:
          1. Map each ID → subword string
          2. Skip [PAD], [BOS], [EOS]
          3. If subword starts with "<W>", prepend a space and remove "<W>"
        """
        tokens = [self.id_to_token.get(idx, "[UNK]") for idx in token_ids]

        text_parts = []
        for token in tokens:
            # Skip structural tokens
            if token in ("[PAD]", "[BOS]", "[EOS]"):
                continue
            if token == "[UNK]":
                text_parts.append("�")  # Standard replacement character
                continue

            # This is the crucial part: check for the space marker first.
            if token.startswith(self.space_token):
                # Add a space, but only if it's not the very first token.
                if text_parts:
                    text_parts.append(" ")
                # Process the rest of the token
                token = token[len(self.space_token):]

            text_parts.append(token)

        return "".join(text_parts)

    def _split_to_chars(self, word: str) -> List[str]:
        # This regex ensures special tokens are treated as whole units
        escape_tokens = list(SPECIAL_TOKENS) + [self.space_token]
        escaped = sorted((re.escape(tok) for tok in escape_tokens), key=len, reverse=True)
        group = "|".join(escaped)
        pattern = rf"(?:{group})|."
        return re.findall(pattern, word)

    def _append_special_tokens(self):
        self.vocab.update(SPECIAL_TOKENS)
        self.vocab.add(self.space_token)
        self.vocab.update(self.special_tokens.keys())

    def _build_corpus_dict(self, all_words):
        # Correctly build the corpus dictionary and initial vocabulary
        word_counts = Counter(tuple(self._split_to_chars(w)) for w in all_words)
        self.corpus_dict = dict(word_counts)
        # Add all unique single characters/symbols to the vocab from the start
        unique_symbols = set(symbol for word_tuple in self.corpus_dict for symbol in word_tuple)
        self.vocab.update(unique_symbols)


    def _create_bi_grams_pre_tokens(self, pre_tokens):
        bigram_counter = self._find_bigrams_in_pre_tokens(pre_tokens)

        # --- Report bi-grams counts ---
        if bigram_counter:
            print("\n--- Bi-gram Frequency Report ---")

            # Get the 5 most common bi-grams
            most_common = bigram_counter.most_common(5)
            print("5 Most Common Bi-grams:")
            for bigram, freq in most_common:
                print(f"  {bigram}: {freq} occurrences")

            # Get the 5 least common bi-grams
            least_common = bigram_counter.most_common()[-5:]
            print("\n5 Least Common Bi-grams:")
            for bigram, freq in reversed(least_common):  # reverse to show lowest first
                print(f"  {bigram}: {freq} occurrences")
            print("---------------------------------\n")
        else:
            print("\n--- No bi-grams found to report. ---\n")
        # --- End of report ---

        frequent_bigrams = {
            pair for pair, count in bigram_counter.items()
            if count >= self.bigrams_freq_threshold
        }

        merged_sentences = []

        for sentence in pre_tokens:
            new_sentence = []
            i = 0
            while i < len(sentence):
                # Try to merge a 3-token sequence first (e.g., word-hyphen-word)
                if i < len(sentence) - 2 and (sentence[i], sentence[i + 1], sentence[i + 2]) in frequent_bigrams:
                    merged = sentence[i] + sentence[i + 1] + sentence[i + 2]
                    new_sentence.append(merged)
                    # i += 3
                # Then try to merge a 2-token sequence
                elif i < len(sentence) - 1 and (sentence[i], sentence[i + 1]) in frequent_bigrams:
                    merged = sentence[i] + sentence[i + 1]
                    new_sentence.append(merged)
                    # i += 2
                else:
                    new_sentence.append(sentence[i])
                i += 1
            merged_sentences.append(new_sentence)
        return merged_sentences

    def _find_bigrams_in_pre_tokens(self, pre_tokens):
        """
        Scan through pre-tokenized sentences and count:
          1. Regular two-word bigrams (word, word)
          2. Three-token sequences connected by hyphens/underscores (word–punct–word)
          3. Three-token sequences for dot-abbreviations (word . word)

        We skip any bigram that involves a determiner or pure punctuation as a word.
        """
        bigram_counter = Counter()

        for sentence in pre_tokens:
            # We need at least two tokens for a bigram, and three for the hyphen/dot cases
            n = len(sentence)
            if n < 2:
                continue

            for i in range(n - 1):
                first = sentence[i]
                second = sentence[i + 1]

                # --- SKIP any pair containing a determiner ---
                if self._check_if_determiner(first) or self._check_if_determiner(second):
                    continue

                if self._check_if_not_ascii(first) or self._check_if_not_ascii(second):
                    continue

                # --- CASE A: hyphen- or underscore-connected word sequences ---
                #    e.g. ["new", "-", "york"]
                if (
                        self._check_if_punctuation(second)
                        and "-" in second
                        and i + 2 < n
                ):
                    third = sentence[i + 2]
                    if not self._check_if_punctuation(first) and not self._check_if_punctuation(third) and not self._check_if_determiner(third) and not self._check_if_not_ascii(third):
                        bigram_counter[(first, second, third)] += 1
                    continue  # move on to next position

                # --- CASE B: regular two-word bigrams (word, word) ---
                if not self._check_if_punctuation(first) and not self._check_if_punctuation(second):
                    bigram_counter[(first, second)] += 1

        return bigram_counter

    def _check_if_punctuation(self, word: str) -> bool:
        # A token is considered punctuation if all its characters are punctuation,
        # ignoring the <W> prefix.
        if word.startswith(self.space_token):
            word = word[len(self.space_token):]
        # Return False if the word is empty after stripping prefix
        return bool(word) and all(c in string.punctuation for c in word)

    def _check_if_determiner(self, word: str) -> bool:
        if word.startswith(self.space_token):
            word = word[len(self.space_token):]
        return word.lower() in DETERMINERS

    def _check_if_not_ascii(self, word: str) -> bool:
        if word.startswith(self.space_token):
            word = word[len(self.space_token):]
        # check for any non-ASCII codepoint
        return any(ord(ch) > 127 for ch in word)

    # --- Core BPE Training Helpers ---

    def _build_merge_cand_dict(self):
        for key, word_freq in self.corpus_dict.items():
            for i in range(len(key) - 1):
                pair = (key[i], key[i + 1])
                curr_freq, pre_tokens_set = self.merge_cand.get(pair, (0, set()))
                pre_tokens_set.add(key)
                self.merge_cand[pair] = (curr_freq + word_freq, pre_tokens_set)

    def _get_best_pair(self):
        if not self.merge_cand:
            return None, 0
        # Filter out pairs that may have a zero frequency after updates
        valid_cands = {p: f for p, (f, s) in self.merge_cand.items() if f > 0}
        if not valid_cands:
            return None, 0
        best_pair = max(valid_cands, key=valid_cands.get)
        return best_pair, valid_cands[best_pair]

    def _update_corpus(self, best_pair):
        _, best_set = self.merge_cand[best_pair]
        merged_token = "".join(best_pair)
        for splited_word in list(best_set):
            if splited_word not in self.corpus_dict: continue
            word_freq = self.corpus_dict.pop(splited_word)

            # Decrement counts for pairs in the old word
            for i in range(len(splited_word) - 1):
                key = (splited_word[i], splited_word[i + 1])
                if key in self.merge_cand:
                    merge_freq, word_lists = self.merge_cand[key]
                    word_lists.discard(splited_word)
                    self.merge_cand[key] = (merge_freq - word_freq, word_lists)

            # Create new word and add it back to corpus
            new_key_list, i = [], 0
            while i < len(splited_word):
                if i < len(splited_word) - 1 and (splited_word[i], splited_word[i + 1]) == best_pair:
                    new_key_list.append(merged_token)
                    i += 2
                else:
                    new_key_list.append(splited_word[i])
                    i += 1
            new_key = tuple(new_key_list)
            self.corpus_dict[new_key] = self.corpus_dict.get(new_key, 0) + word_freq

            # Increment counts for pairs in the new word
            for i in range(len(new_key) - 1):
                pair = (new_key[i], new_key[i + 1])
                curr_freq, pre_tokens_set = self.merge_cand.get(pair, (0, set()))
                pre_tokens_set.add(new_key)
                self.merge_cand[pair] = (curr_freq + word_freq, pre_tokens_set)
