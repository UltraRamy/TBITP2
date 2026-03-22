# Search Engine from Scratch

A search engine built from scratch using only Python standard libraries. Implements indexing, compression, retrieval, and evaluation components from the ground up.

---

## Project Structure

```
├── collection/        # Document corpus (text files organized in subdirectories)
├── index/             # Output directory for the generated inverted index
├── tmp/               # Temporary storage during indexing
├── bsbi.py            # BSBI indexing + TF-IDF and BM25 retrieval (including WAND)
├── spimi.py           # SPIMI indexing with Trie-based dictionary
├── compression.py     # Postings list compression (Standard, VBE, Elias-Gamma)
├── index.py           # Inverted index read/write logic
├── util.py            # IdMap, Trie, TrieIdMap, merge utilities
├── evaluation.py      # Evaluation metrics: RBP, DCG, NDCG, AP
├── search.py          # Example retrieval script
├── qrels.txt          # Relevance judgments for evaluation
└── queries.txt        # 30 queries for evaluation
```

---

## How to Run

### 1. Index the collection (BSBI)

```bash
py bsbi.py
```

Or using SPIMI:

```bash
py spimi.py
```

### 2. Run example queries

```bash
py search.py
```

### 3. Evaluate retrieval quality

```bash
py evaluation.py
```

---

## Features

### 1. Bit-Level Compression — Elias-Gamma Encoding (`compression.py`)

In addition to the existing Variable-Byte Encoding (VBE), **Elias-Gamma** encoding is implemented as a bit-level compression scheme.

For a positive integer `n`, Elias-Gamma encodes it as:
- `k = floor(log2(n))` zero bits
- a `1` bit separator
- the last `k` bits of `n` in binary

Postings lists are gap-encoded before compression. A 4-byte count prefix is prepended so the decoder knows when to stop.

```python
from compression import EliasGammaPostings

encoded = EliasGammaPostings.encode([34, 67, 89, 454])
decoded = EliasGammaPostings.decode(encoded)
```

To use Elias-Gamma instead of VBE, change `postings_encoding` in `bsbi.py`:

```python
BSBI_instance = BSBIIndex(data_dir='collection',
                           postings_encoding=EliasGammaPostings,
                           output_dir='index')
```

---

### 2. BM25 Scoring (`bsbi.py`)

**Okapi BM25** is implemented alongside TF-IDF. Document lengths and average document length are precomputed during indexing and stored in the index metadata.

```
IDF(t)       = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
TF_norm(t,D) = tf(t,D) * (k1+1) / (tf(t,D) + k1*(1 - b + b*|D|/avgdl))
score(D,Q)   = Σ IDF(t) * TF_norm(t,D)
```

Default parameters: `k1=1.2`, `b=0.75`.

```python
results = BSBI_instance.retrieve_bm25("query terms here", k=10)
```

---

### 3. Evaluation Metrics (`evaluation.py`)

Four metrics are implemented and evaluated across 30 queries:

| Metric | Description |
|--------|-------------|
| **RBP** | Rank-Biased Precision (p=0.8) |
| **DCG** | Discounted Cumulative Gain |
| **NDCG** | Normalized DCG |
| **AP** | Average Precision |

Running `evaluation.py` prints a comparison table for both TF-IDF and BM25:

```
=============================================
Metric              TF-IDF            BM25
=============================================
RBP                 0.5980          0.6317
DCG                 5.5935          5.7446
NDCG                0.7827          0.7931
MAP                 0.4439          0.4779
=============================================
```

---

### 4. WAND Top-K Retrieval (`bsbi.py`)

**WAND (Weak AND)** is implemented to avoid scoring every document with BM25. Each term stores a precomputed upper bound (`max_tf`) in the index. At retrieval time, WAND uses these bounds to skip documents that cannot enter the top-K.

```python
results = BSBI_instance.retrieve_bm25_wand("query terms here", k=10)
```

WAND produces identical results to full BM25 but skips non-competitive documents.

---

### 5. SPIMI Indexing with Trie Dictionary (`spimi.py`, `util.py`)

**SPIMI (Single-Pass In-Memory Indexing)** is implemented as an alternative to BSBI. Unlike BSBI which requires a global term ID mapping during indexing, SPIMI builds a `term_string → postings` dictionary directly in memory. When the block size is reached, the dictionary is sorted alphabetically and flushed to disk. Term IDs are assigned only during the final merge.

SPIMI uses a **Trie** (`TrieIdMap`) instead of a Python dict for forward string-to-ID mapping, providing prefix-sharing memory structure.

```python
from spimi import SPIMIIndex
from compression import VBEPostings

spimi = SPIMIIndex(data_dir='collection',
                   postings_encoding=VBEPostings,
                   output_dir='index')
spimi.index()

results = spimi.retrieve_bm25("query terms here", k=10)
```

---

## Requirements

- Python 3.x
- `tqdm` (`pip install tqdm`)
