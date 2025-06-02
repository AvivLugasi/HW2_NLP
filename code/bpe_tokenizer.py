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

from patterns_and_dicts import SPECIAL_TOKENS, DETERMINERS


class BPETokenizer(BaseTokenizer, ABC):
    def __init__(self,
                 vocab_size: int = 5000,
                 normalizer: Normalizer = None,
                 pre_tokenizer: PreTokenizer = None,
                 enable_bigrams: bool = True,
                 bigrams_freq_threshold: int = 10,
                 merge_freq_threshold: int = 1):
        """
        A Byte-Pair Encoding tokenizer that extends BaseTokenizer.
        Uses "<W>" as the word-begin marker to stay in sync with PreTokenizer.
        """
        # Initialize BaseTokenizer (sets up [PAD],[UNK],[BOS],[EOS])
        super().__init__()

        # BPE-specific containers
        self.pre_tokens: Dict[tuple[str], int] = {}
        self.merge_cand: Dict[Tuple[str, str], int] = {}
        self.rules: Dict[str, int] = {}
        self.vocab: Set[str] = set()

        # append special tokens
        self._append_special_tokens()

        # How many merge-iterations to run
        self.vocab_size = vocab_size

        # Marker for "beginning of word" (must match PreTokenizer's marker)
        self.space_token = "<W>"

        # Normalizer
        if normalizer is None:
            normalizer = Normalizer(
                unicode_normalization='NFKD',
                lower_case="TITLE CASE + STOP WORDS",
                remove_accents=True,
                expand_contractions=True,
                replace_urls=True,
                replace_usernames=True,
                replace_hashtag=True,
                replace_html_tags=True
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

        # 3) Build tokenized_corpus and pair_positions
        all_words = [w for line_tokens in pre_tokenized_sentences for w in line_tokens]

        self._build_pre_tokens_dict(all_words=all_words)
        logging.info(f"==== Built tokenized_corpus (total words = {len(self.pre_tokens)}) ====\n")

        merges_done = 0
        # 4) Perform merges up to vocab_size
        while len(self.vocab) < self.vocab_size:
            # 4a) calculate the frequencies of each possible merge and return the pair with the highest
            best_pair, freq = self._build_merge_cand_dict()
            if best_pair is None or freq < self.merge_freq_threshold:
                print(best_pair)
                print(freq)
                break
            # define the merge
            merged_symbol = ''.join(best_pair)

            # 4b) remove the pair from the merge cand list
            del self.merge_cand[best_pair]
            # 4c) update the pre tokens dict
            self._update_pre_tokens(best_pair)
            # 4d) update rule and vocab
            self.rules[merged_symbol] = freq
            self.vocab.add(merged_symbol)
            merges_done += 1
            logging.info(f"--- Merge #{merges_done}: {best_pair} → '{merged_symbol}' (freq={freq})")
            logging.info("Remaining distinct pairs will be recomputed next iteration…\n")

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
        Convert a raw text string → list of token IDs via BPE:
          1. Normalize
          2. Pre-tokenize in inference mode
          3. For each token, build ["<W>", c1, c2, ...] or list(word) if no "<W>"
          4. Apply each merge from self.rules in order
          5. Map each resulting subword → ID
        """
        normalized = self.normalizer.normalize_text(text=text)
        self.pre_tokenizer.train_mode = False
        pre_tokens = self.pre_tokenizer.pre_tokenize_str(text=normalized)
        splits = [self._split_to_chars(word) for word in pre_tokens]
        splits = list(itertools.chain.from_iterable(splits))
        tokens = self._apply_merges_on_words(splits)
        print(tokens)
        return [self.token_to_id[token] for token in tokens]

    def _apply_merges_on_words(self, words):
        possible_to_merge = True
        while possible_to_merge:
            word_merge_cands_pairs = [(words[i], words[i + 1], i) for i in range(len(words) - 1)]
            for pair in word_merge_cands_pairs:
                if "".join(pair[:2]) in self.rules.keys():
                    merge = "".join(pair[:2])
                    index = pair[-1]
                    words = words[:index] + [merge] + words[index + 2:]
                    possible_to_merge = True
                    break
                else:
                    possible_to_merge = False
        return words

    def decode(self, token_ids: List[int]) -> str:
        """
        Convert a list of token IDs back to a text string:
          1. Map each ID → subword string
          2. Skip [PAD], [BOS], [EOS]
          3. If subword starts with "<W>", prepend a space and remove "<W>"
        """
        subwords: List[str] = []
        for idx in token_ids:
            subwords.append(self.id_to_token.get(idx, "[UNK]"))
        reconstructed = ""

        for i, sw in enumerate(subwords):
            if sw in ("[PAD]", "[BOS]", "[EOS]"):
                continue
            reconstructed += sw

        reconstructed = re.sub(re.escape(self.space_token), '', reconstructed, count=1)
        # Replace all remaining occurrences with a space
        reconstructed = re.sub(re.escape(self.space_token), ' ', reconstructed)

        return reconstructed.strip()

    def initial_spliter(self, text: str) -> List[List[str]]:
        """
        Given a raw text string, normalize & pre-tokenize, then return a nested list
        of symbol-lists for each token:
          e.g. "Hello world" → [["<W>","H","e","l","l","o"], ["<W>","w","o","r","l","d"]]
        """
        normalized = self.normalizer.normalize_text(text=text)
        self.pre_tokenizer.train_mode = True
        pre_tokens = self.pre_tokenizer.pre_tokenize_str(text=normalized)

        result: List[List[str]] = []
        for tok in pre_tokens:
            if tok.startswith(self.space_token):
                symbols = [self.space_token] + list(tok[len(self.space_token):])
            else:
                symbols = list(tok)
            result.append(symbols)
        return result

    def _get_stats(self, ids: List[str]) -> None:
        """
        Count adjacent pairs in one sequence of symbols (ids)
        and update self.merge_cand accordingly.
        """
        for pair in zip(ids, ids[1:]):
            self.merge_cand[pair] = self.merge_cand.get(pair, 0) + 1

    def _split_to_chars(self, word: str):
        space_to_add = None
        if word.startswith(self.space_token):
            space_to_add = [self.space_token]
            word = word[len(self.space_token):]
        splitted_word = [word] if word in SPECIAL_TOKENS else list(word)
        return space_to_add + splitted_word if space_to_add else splitted_word

    def _append_special_tokens(self):
        self.vocab.update(SPECIAL_TOKENS)

    def _build_pre_tokens_dict(self, all_words):
        for w in all_words:
            # get the word splitted to its most basic components (special tokens and individual chars)
            splitted_w = self._split_to_chars(w)
            # update vocab
            self.vocab.update(splitted_w)
            splited_w_as_key = tuple(splitted_w)
            if splited_w_as_key not in self.pre_tokens:
                self.pre_tokens[splited_w_as_key] = 0
            freq = self.pre_tokens[splited_w_as_key]
            self.pre_tokens[splited_w_as_key] = freq + 1

    def _create_bi_grams_pre_tokens(self, pre_tokens):
        bigram_counter = self._find_bigrams_in_pre_tokens(pre_tokens)

        frequent_bigrams = self._return_most_common_bigrams(bigram_counter)

        # merge pre tokens that appear the most in the pre tokens lists
        merged_sentences = []

        for sentence in pre_tokens:
            new_sentence = []
            i = 0
            while i < len(sentence):
                if i < len(sentence) - 1 and (sentence[i], sentence[i + 1]) in frequent_bigrams:
                    merged = sentence[i] + sentence[i + 1]
                    new_sentence.append(merged)
                    i += 2
                elif i < len(sentence) - 1 and self._check_if_punctuation(sentence[i + 1]) \
                        and (
                        i < len(sentence) - 2 and (sentence[i], sentence[i + 1], sentence[i + 2]) in frequent_bigrams):
                    merged = sentence[i] + sentence[i + 1] + sentence[i + 2]
                    new_sentence.append(merged)
                    i += 3
                else:
                    new_sentence.append(sentence[i])
                    i += 1
            merged_sentences.append(new_sentence)
        return merged_sentences

    def _find_bigrams_in_pre_tokens(self, pre_tokens):
        bigram_counter = Counter()

        for sentence in pre_tokens:
            for i in range(len(sentence) - 1):
                # avoid bi grams as word + punctuation, avoid stop words bi grams
                if not self._check_if_punctuation(sentence[i]) and not self._check_if_determiner(
                        sentence[i]) and not self._check_if_determiner(sentence[i + 1]):
                    if self._check_if_punctuation(sentence[i + 1]):
                        if "-" in sentence[i + 1] or "_" in sentence[i + 1]:
                            if i + 2 < (len(sentence)) and not self._check_if_punctuation(
                                    sentence[i + 2]) and not self._check_if_determiner(sentence[i + 2]):
                                bigram_counter[(sentence[i], sentence[i + 1], sentence[i + 2])] += 1
                    else:
                        bigram_counter[(sentence[i], sentence[i + 1])] += 1

        return bigram_counter

    def _check_if_punctuation(self, word):
        flag = False
        if word in string.punctuation or word == self.space_token:
            flag = True
        if word.startswith(self.space_token):
            flag = True
            for char in word[len(self.space_token):]:
                if char not in string.punctuation:
                    flag = False
                    break
        return flag

    def _check_if_determiner(self, word):
        if word.startswith(self.space_token):
            word = word[len(self.space_token):]
        return word.lower() in DETERMINERS

    def _return_most_common_bigrams(self, bigram_counter):
        return {
            pair: count
            for pair, count in bigram_counter.items()
            if count >= self.bigrams_freq_threshold
        }

    def _build_merge_cand_dict(self):
        most_freq_pair = None
        most_freq_pair_freq = 0
        for key in self.pre_tokens.keys():
            word_freq = self.pre_tokens[key]
            word_merge_cands_pairs = [(key[i], key[i + 1]) for i in range(len(key) - 1)]
            for pair in word_merge_cands_pairs:
                if pair not in self.merge_cand:
                    self.merge_cand[pair] = 0
                prev_freq = self.merge_cand[pair]
                self.merge_cand[pair] = prev_freq + word_freq
                if most_freq_pair_freq < self.merge_cand[pair]:
                    most_freq_pair = pair
                    most_freq_pair_freq = self.merge_cand[pair]
        return most_freq_pair, most_freq_pair_freq

    def _update_pre_tokens(self, best_pair):
        merged_symbol = ''.join(best_pair)
        updates = {}

        for key, freq in list(self.pre_tokens.items()):
            new_key = []
            i = 0
            while i < len(key):
                # Check if best_pair matches at position i
                if i < len(key) - 1 and key[i] == best_pair[0] and key[i + 1] == best_pair[1]:
                    new_key.append(merged_symbol)
                    i += 2  # skip the next symbol as it's part of the merged pair
                else:
                    new_key.append(key[i])
                    i += 1
            new_key = tuple(new_key)
            if new_key not in updates:
                updates[new_key] = 0
            updates[new_key] += freq

        self.pre_tokens = updates  # Replace instead of update to avoid stale keys


import os

domain_file = "../data/domain_1_sample.txt"
# domain_file = "../data/domain_1_train.txt"
# domain_file = "../data/small_test_data.txt"
output_dir = "../tokenizers"

os.makedirs(output_dir, exist_ok=True)

# Read domain data
print(f"Reading domain data from {domain_file}")
with open(domain_file, 'r', encoding='utf-8') as f:
    texts = f.readlines()

print(f"Read {len(texts)} lines of text")

# Initialize and train tokenizer
print(f"Training BPE tokenizer with vocab size {5000}")
tokenizer = BPETokenizer(vocab_size=5000,
                         merge_freq_threshold=5)
#
# # # 1) Normalize all texts
# # normalized_texts = normalize_text_file(
# #     normalizer=tokenizer.normalizer,
# #     batch_of_text=(texts, 0)
# # )
# # logging.info("==== Done Normalizing ====\n")
#
# # with open("../data/domain_1_sample_normalized.txt", "w", encoding="utf8") as f_out:
# #     for sentence_tokens in normalized_texts:
# #         f_out.write(sentence_tokens+"\n")
#
# # 2) Pre-tokenize in train_mode
# # tokenizer.pre_tokenizer.train_mode = True
# # pre_tokenized_sentences = pre_tokenize_text_file(
# #     pre_tokenizer=tokenizer.pre_tokenizer,
# #     batch_of_text=(normalized_texts, 0)
# # )
# # logging.info("==== Done Pre-tokenize ====\n")
# # with open("../data/domain_1_sample_pre_tokenized.txt", "w", encoding="utf8") as f_out:
# #     for sentence_tokens in pre_tokenized_sentences:
# #         f_out.write(" ".join(sentence_tokens) + "\n")
#
import time

start_time = time.time()
tokenizer.train(texts)
end_time = time.time()

print(f"train took {end_time - start_time:.4f} seconds to run.")
#
# # tokenizer.save("../tokenizers/sample_tokenizer.pkl")
# # # Save the tokenizer
# # output_path = os.path.join(output_dir, "tokenizer.pkl")
# # print(f"Saving tokenizer to {output_path}")
# # tokenizer.save(output_path)
# # print(f"Tokenizer trained with {tokenizer.get_vocab_size()} tokens")
# #
# # # Test the tokenizer on a sample
if texts:
    # sample_text = texts[0].strip()
    sample_text = "I went to empire state building yesterday and it was amazing."
    print("\nExample encoding/decoding:")
    print(f"Original text: {sample_text}")

    encoded = tokenizer.encode(sample_text)
    print(encoded)
    print(f"Encoded: {encoded[:50]}{'...' if len(encoded) > 50 else ''}")

    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")
