# bpe_tokenizer.py

from abc import ABC
from typing import List, Tuple, Dict, Set
import pickle

from base_tokenizer import BaseTokenizer
from normalizer import Normalizer, normalize_text_file
from pre_tokenizer import PreTokenizer, pre_tokenize_text_file

import heapq
from collections import defaultdict


class BPETokenizer(BaseTokenizer, ABC):
    def __init__(self,
                 vocab_size: int = 5000,
                 normalizer: Normalizer = None,
                 pre_tokenizer: PreTokenizer = None):
        """
        A Byte-Pair Encoding tokenizer that extends BaseTokenizer.
        Uses "<W>" as the word-begin marker to stay in sync with PreTokenizer.
        """
        # Initialize BaseTokenizer (sets up [PAD],[UNK],[BOS],[EOS])
        super().__init__()

        # BPE-specific containers
        self.merge_cand: Dict[Tuple[str, str], int] = {}
        self.rules: List[Tuple[str, str]] = []
        self.vocab: Set[str] = set()

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


    def train(self, texts: List[str]) -> None:
        """
        Incremental‐update BPE training with bounds check to avoid out-of-range.
        1. Normalize & pre-tokenize the corpus.
        2. Build tokenized_corpus: List[List[str]] of symbol‐lists.
        3. Build pair_positions: map each (sym_i, sym_{i+1}) → set of (word_idx, pos).
        4. Build a max‐heap of (−frequency, pair).
        5. Repeat up to self.vocab_size merges:
           a. Pop highest‐frequency pair (A, B) from heap (skip stale entries).
           b. For each recorded location (word_idx, pos), check bounds and merge (A,B)→A+B.
           c. Update neighbor pairs locally and push updated frequencies into the heap.
        6. After merges, collect all remaining symbols into self.vocab and assign IDs (preserving specials).
        """
        print(f"==== Starting incremental BPE training for {len(texts)} lines, up to {self.vocab_size} merges ====\n")

        # 1) Normalize all texts
        normalized_texts = normalize_text_file(
            normalizer=self.normalizer,
            batch_of_text=(texts, 0)
        )
        print("==== Done Normalizing ====\n")

        # 2) Pre-tokenize in train_mode
        self.pre_tokenizer.train_mode = True
        pre_tokenized_sentences = pre_tokenize_text_file(
            pre_tokenizer=self.pre_tokenizer,
            batch_of_text=(normalized_texts, 0)
        )
        print("==== Done Pre-tokenize ====\n")

        # 3) Build tokenized_corpus and pair_positions
        tokenized_corpus: List[List[str]] = []
        pair_positions: Dict[Tuple[str, str], Set[Tuple[int, int]]] = defaultdict(set)

        for line_tokens in pre_tokenized_sentences:
            for word in line_tokens:
                # Convert word → symbols, e.g. "<W>hello" → ["<W>", "h", "e", "l", "l", "o"]
                if word.startswith(self.space_token):
                    symbols = [self.space_token] + list(word[len(self.space_token):])
                else:
                    symbols = list(word)

                word_idx = len(tokenized_corpus)
                tokenized_corpus.append(symbols)

                # Register adjacent pairs in this symbol list
                for i in range(len(symbols) - 1):
                    pair = (symbols[i], symbols[i + 1])
                    pair_positions[pair].add((word_idx, i))

        print("==== Done Build tokenized_corpus and pair_positions ====\n")

        # 4) Build max‐heap of (−frequency, pair)
        heap: List[Tuple[int, Tuple[str, str]]] = []
        for pair, locs in pair_positions.items():
            freq = len(locs)
            if freq > 0:
                heapq.heappush(heap, (-freq, pair))

        # 5) Perform merges incrementally
        self.rules = []
        merges_done = 0

        while merges_done < self.vocab_size and heap:
            neg_freq, best_pair = heapq.heappop(heap)
            freq = -neg_freq

            # Check for stale entry
            actual_freq = len(pair_positions.get(best_pair, set()))
            if actual_freq != freq:
                if actual_freq > 0:
                    heapq.heappush(heap, (-actual_freq, best_pair))
                continue

            # If no useful merges remain, break
            if freq < 1:
                print(f"Iteration {merges_done + 1}: highest freq < 1 → stopping early.\n")
                break

            merges_done += 1
            self.rules.append(best_pair)
            A, B = best_pair
            merged_symbol = A + B

            print(f"--- Merge #{merges_done}: {best_pair} → '{merged_symbol}' (freq={freq})")

            # 5a) For each recorded location of (A, B), merge in tokenized_corpus
            locations = list(pair_positions[best_pair])
            for (word_idx, pos) in locations:
                symbols = tokenized_corpus[word_idx]

                # **Bounds check**: ensure pos+1 is still valid
                if pos + 1 >= len(symbols):
                    # This location is stale (the word was shortened by prior merges)
                    continue

                if symbols[pos] != A or symbols[pos + 1] != B:
                    # Something changed here already—skip
                    continue

                # Perform the actual merge: replace A, B → merged_symbol
                symbols[pos] = merged_symbol
                del symbols[pos + 1]

                # Remove this location from pair_positions[(A, B)]
                pair_positions[best_pair].discard((word_idx, pos))

                # 5b) Update neighbor pairs around pos

                # Left neighbor: (X, A) → now (X, merged_symbol)
                left_i = pos - 1
                if left_i >= 0:
                    X = symbols[left_i]
                    old_left = (X, A)
                    if (word_idx, left_i) in pair_positions.get(old_left, set()):
                        pair_positions[old_left].discard((word_idx, left_i))
                        if old_left in pair_positions and len(pair_positions[old_left]) > 0:
                            heapq.heappush(heap, (-len(pair_positions[old_left]), old_left))

                    new_left = (X, merged_symbol)
                    pair_positions[new_left].add((word_idx, left_i))
                    heapq.heappush(heap, (-len(pair_positions[new_left]), new_left))

                # Right neighbor: (B, Y) → now (merged_symbol, Y)
                right_i = pos + 1
                if right_i < len(symbols):
                    Y = symbols[right_i]
                    old_right = (B, Y)
                    if (word_idx, pos) in pair_positions.get(old_right, set()):
                        pair_positions[old_right].discard((word_idx, pos))
                        if old_right in pair_positions and len(pair_positions[old_right]) > 0:
                            heapq.heappush(heap, (-len(pair_positions[old_right]), old_right))

                    new_right = (merged_symbol, Y)
                    pair_positions[new_right].add((word_idx, pos))
                    heapq.heappush(heap, (-len(pair_positions[new_right]), new_right))

            # 5c) If (A, B) has no remaining positions, delete it
            if best_pair in pair_positions and not pair_positions[best_pair]:
                del pair_positions[best_pair]

            print(f"Remaining distinct pairs in heap: {len(heap)}\n")

        # 6) Build final vocabulary from remaining symbols
        self.vocab.clear()
        for seq in tokenized_corpus:
            for sym in seq:
                self.vocab.add(sym)

        # 7) Assign IDs (preserve [PAD],[UNK],[BOS],[EOS])
        next_id = max(self.token_to_id.values()) + 1  # should start at 4
        for subword in sorted(self.vocab):
            if subword not in self.token_to_id:
                self.token_to_id[subword] = next_id
                self.id_to_token[next_id] = subword
                next_id += 1

        print(f"==== Finished BPE training: {merges_done} merges done, final vocab size = {len(self.vocab) + 4} (including special tokens) ====\n")

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

        token_ids: List[int] = []
        # (Optional) token_ids.append(self.token_to_id["[BOS]"])

        for word in pre_tokens:
            if word.startswith(self.space_token):
                symbols: List[str] = [self.space_token] + list(word[len(self.space_token):])
            else:
                symbols = list(word)

            # Apply merges
            for (A, B) in self.rules:
                i = 0
                new_symbols: List[str] = []
                while i < len(symbols):
                    if i < len(symbols) - 1 and symbols[i] == A and symbols[i + 1] == B:
                        new_symbols.append(A + B)
                        i += 2
                    else:
                        new_symbols.append(symbols[i])
                        i += 1
                symbols = new_symbols

            # Map subwords → IDs
            for subword in symbols:
                if subword in self.token_to_id:
                    token_ids.append(self.token_to_id[subword])
                else:
                    token_ids.append(self.token_to_id["[UNK]"])

        # (Optional) token_ids.append(self.token_to_id["[EOS]"])
        return token_ids

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
        for sw in subwords:
            if sw in ("[PAD]", "[BOS]", "[EOS]"):
                continue
            if sw.startswith(self.space_token):
                reconstructed += " " + sw[len(self.space_token):]
            else:
                reconstructed += sw

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
