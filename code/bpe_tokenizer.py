
from abc import ABC
from typing import List, Tuple, Dict, Set

from base_tokenizer import BaseTokenizer
from normalizer import Normalizer, normalize_text_file
from pre_tokenizer import PreTokenizer, pre_tokenize_text_file
from collections import Counter


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
        all_words = [w for line_tokens in pre_tokenized_sentences for w in line_tokens]
        tokenized_corpus = [
            ([self.space_token] + list(w[len(self.space_token):]))
            if w.startswith(self.space_token)
            else list(w)
            for w in all_words
        ]
        print(f"==== Built tokenized_corpus (total words = {len(tokenized_corpus)}) ====\n")


        self.rules = []
        merges_done = 0

        # 4) Perform merges up to vocab_size
        while merges_done < self.vocab_size:
            # 4a) Count all adjacent pairs in the entire corpus
            pair_counts = Counter()
            for symbols in tokenized_corpus:
                pair_counts.update(zip(symbols, symbols[1:]))

            if not pair_counts:
                # No pairs left → stop
                break

            # 4b) Select the most common pair
            best_pair, freq = pair_counts.most_common(1)[0]
            if freq < 1:
                break

            merges_done += 1
            self.rules.append(best_pair)
            A, B = best_pair
            merged_symbol = A + B
            print(f"--- Merge #{merges_done}: {best_pair} → '{merged_symbol}' (freq={freq})")

            # 4c) Rebuild each word that contains (A, B)
            for idx, symbols in enumerate(tokenized_corpus):
                # Quick check: skip if A or B aren't in this word at all
                if A not in symbols or B not in symbols:
                    continue

                new_symbols: List[str] = []
                i = 0
                did_merge = False
                while i < len(symbols):
                    if i < len(symbols) - 1 and symbols[i] == A and symbols[i + 1] == B:
                        new_symbols.append(merged_symbol)
                        i += 2
                        did_merge = True
                    else:
                        new_symbols.append(symbols[i])
                        i += 1

                if did_merge:
                    tokenized_corpus[idx] = new_symbols

            print("Remaining distinct pairs will be recomputed next iteration…\n")

        # 5) After merging, collect final vocabulary from all symbols
        self.vocab.clear()
        for symbols in tokenized_corpus:
            for sym in symbols:
                self.vocab.add(sym)

        # 6) Assign token IDs to each subword (preserving [PAD],[UNK],[BOS],[EOS])
        next_id = max(self.token_to_id.values()) + 1
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
