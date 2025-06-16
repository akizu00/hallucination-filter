from collections import Counter
import pandas as pd
from typing import List
import math


def extract_bigrams(seq: List[int]) -> List[tuple]:
    return [(seq[i], seq[i + 1]) for i in range(len(seq) - 1)]


def build_bigram_matrix(seq: List[int]) -> pd.DataFrame:
    bigrams = extract_bigrams(seq)
    counts = Counter(bigrams)

    row_labels = sorted(set(x[0] for x in counts))
    col_labels = sorted(set(x[1] for x in counts))

    matrix = pd.DataFrame(0, index=row_labels, columns=col_labels)

    for (r, c), count in counts.items():
        matrix.loc[r, c] = count

    return counts, matrix

#calculate entropy
def entropy1(seq: List[int]) -> float:
    total = len(seq)
    counter = Counter(seq)
    probs = [count / total for count in counter.values()]
    return -sum(p * math.log2(p) for p in probs)


def entropy2(seq: List[int]) -> float:
    bigrams = extract_bigrams(seq)
    total = len(bigrams)
    counter = Counter(bigrams)
    probs = [count / total for count in counter.values()]
    return -sum(p * math.log2(p) for p in probs)

if __name__ == "__main__":
    sequences = [
        [1, 2, 3, 2, 3],
        [1, 2, 3, 2, 1, 5, 6],
        [1, 2, 3, 4, 2, 2, 3, 4, 5],
        [1, 2, 1]
    ]

    for idx, seq in enumerate(sequences, start=1):
        print(f"\nSequence {idx}: {seq}")
        e1 = entropy1(seq)
        e2 = entropy2(seq)
        print(f"Entropy1 (unigram): {e1:.4f}")
        print(f"Entropy2 (bigram):  {e2:.4f}")

        # Extract bigrams pairs
        pairs = extract_bigrams(seq)
        print(f"Extracted Bigrams: {pairs}")

        # Form matrix
        counts, matrix = build_bigram_matrix(seq)

        # Count how many pairs
        print("Count how many Bigram pairs:")
        for pair, count in counts.items():
            print(f"  {pair} >> {count}")

        # Show matrix
        print("Bigram Frequency Matrix:")
        print(matrix)
