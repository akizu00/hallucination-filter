from collections import Counter
import math
import re

#normalize the text.file
def normalize(text: str) -> str:
    text = text.lower()  # lowercase
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    return text

def ngram_entropy(text: str, n: int = 2) -> float:
    words = text.strip().split()
    if len(words) < n:
        return 0.0

    ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    ngram_counts = Counter(ngrams)
    total = sum(ngram_counts.values())

    entropy = 0.0
    for count in ngram_counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy

def process_file(file_path, threshold=2.0, n=2):
    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f, start=1):      # iterate line by line
            text = normalize(line.strip())
            ent  = ngram_entropy(text, n)
            status = "HALLUCINATED" if ent < threshold else "OK"
            print(f"[Line {idx:03d}] {n}-gram Entropy = {ent:.2f} → {status}\n  ▶ {line.strip()}")

if __name__ == "__main__":
    for n in [2, 3, 4]:
        process_file("data/text.txt", threshold=2.0, n=n)  # Use n=3 for trigram

