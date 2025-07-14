from typing import Literal
from normalizer import Normalizer
from pre_tokenizer import PreTokenizer
import os
from utils import *
from bpe_tokenizer import BPETokenizer
from combiner import DatasetBalancer


def generate_normalizer(domain: Literal[1, 2, 3]):
    if domain == 1:
        return Normalizer(
            unicode_normalization=None,
            lower_case="TITLE CASE + STOP WORDS",
            remove_needless_ws = True,
            remove_repeated_letters=True,
            remove_needless_punctuation=True
        )
    elif domain == 2:
        return Normalizer(
            unicode_normalization=None,
            lower_case=None,
            remove_needless_ws=True,
            remove_repeated_letters=False,
            remove_needless_punctuation=False
        )
    else:
        return Normalizer(
            unicode_normalization=None,
            lower_case=None,
            remove_needless_ws=True,
            remove_repeated_letters=False,
            remove_needless_punctuation=False
        )


def generate_pre_tokenizer(domain: Literal[1, 2, 3]):
    if domain == 1:
        return PreTokenizer(
            split_punctuation=True
        )
    elif domain == 2:
        return PreTokenizer(
            split_punctuation=True
        )
    else:
        return PreTokenizer(
            split_punctuation=True
        )


def generate_tokenizer(domain: Literal[1, 2, 3]):
    logging.info(f"Generate tokenizer for domain-{domain}")
    normalizer = generate_normalizer(domain)
    pre_tokenizer = generate_pre_tokenizer(domain)

    if domain == 1:
        vocab_size = 5000
    else:
        vocab_size = 10000

    if domain == 3:
        domain_file = f"../data_for_tokenizer_3/domain_{domain}_train.txt"
    else:
        domain_file = f"../data/domain_{domain}_train.txt"

    output_dir = "../trained_tokenizers"
    tokenizer_file = f"tokenizer_{domain}.pkl"
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Reading domain data from {domain_file}")
    with open(domain_file, 'r', encoding='utf-8') as f:
        texts = f.readlines()

    logging.info(f"Training BPE tokenizer with vocab size {vocab_size}")
    tokenizer = BPETokenizer(vocab_size=vocab_size,
                             normalizer=normalizer,
                             pre_tokenizer=pre_tokenizer)
    tokenizer.train(texts)
    output_path = os.path.join(output_dir, tokenizer_file)
    logging.info(f"Saving tokenizer to {output_path}")
    tokenizer.save(output_path)
    logging.info(f"Tokenizer trained with {tokenizer.get_vocab_size()} tokens")


def main():
    for i in [1, 2, 3]:
        generate_tokenizer(domain=i)



if __name__ == "__main__":
    main()