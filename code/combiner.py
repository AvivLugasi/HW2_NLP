import random

class DatasetBalancer:
    def __init__(self, domain1_path, domain2_path, output_path, seed=42):
        self.domain1_path = domain1_path
        self.domain2_path = domain2_path
        self.output_path = output_path
        self.seed = seed

    def load_sentences(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

    def save_sentences(self, sentences, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            for sentence in sentences:
                f.write(sentence + '\n')

    def create_balanced_dataset(self):
        # Load data
        domain1_sentences = self.load_sentences(self.domain1_path)
        domain2_sentences = self.load_sentences(self.domain2_path)

        # Undersample domain1
        random.seed(self.seed)
        sampled_domain1 = random.sample(domain1_sentences, len(domain2_sentences))

        # Combine and shuffle
        combined = sampled_domain1 + domain2_sentences
        random.shuffle(combined)

        # Save to output
        self.save_sentences(combined, self.output_path)
        print(f"Saved {len(combined)} balanced sentences to '{self.output_path}'")

    def run(self):
        self.create_balanced_dataset()
