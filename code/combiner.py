import random

def load_sentences(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def save_sentences(sentences, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            f.write(sentence + '\n')

def create_balanced_dataset(domain1_path, domain2_path, output_path, seed=42):
    # Load data
    domain1_sentences = load_sentences(domain1_path)
    domain2_sentences = load_sentences(domain2_path)

    # Undersample domain1
    random.seed(seed)
    sampled_domain1 = random.sample(domain1_sentences, len(domain2_sentences))

    # Combine datasets
    combined = sampled_domain1 + domain2_sentences
    random.shuffle(combined)

    # Save to output file
    save_sentences(combined, output_path)
    print(f"Saved {len(combined)} balanced sentences to '{output_path}'")


def main():
    domain1_path = 'data/domain_1_train.txt'
    domain2_path = 'data/domain_2_train.txt'
    output_path = 'data/combined_balanced_train.txt'

    create_balanced_dataset(domain1_path, domain2_path, output_path)


if __name__ == '__main__':
    main()