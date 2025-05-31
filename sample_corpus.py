# sample_corpus.py

import random

input_path  = "data/domain_1_train.txt"
output_path = "data/domain_1_sample.txt"
sample_frac = 0.001  # 10%

# 1) Read all lines into memory (OK if file isn’t too huge on your machine)
with open(input_path, "r", encoding="utf8") as f:
    lines = f.readlines()

n_total = len(lines)
n_sample = int(n_total * sample_frac)

# 2) Randomly sample without replacement
random.seed(42)  # for reproducibility
sampled_lines = random.sample(lines, n_sample)

# 3) Write out the sampled subset
with open(output_path, "w", encoding="utf8") as f_out:
    f_out.writelines(sampled_lines)

print(f"Sampled {n_sample} / {n_total} lines → {output_path}")
