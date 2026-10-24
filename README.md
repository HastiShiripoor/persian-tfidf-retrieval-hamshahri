# Persian Text Retrieval with TF-IDF (Hamshahri Corpus)

A from-scratch **TF-IDF information retrieval system for Persian news text**. It indexes the Hamshahri corpus, ranks documents for a query by cosine similarity, and evaluates the ranking with Precision@k, Recall@k, F1@k and MAP against the provided relevance judgements.

> University course project. Author: `<Hasti Shiripoor / https://github.com/HastiShiripoor>` 

## Overview

Given a Persian query, the system:

1. normalizes, tokenizes, filters stop words and stems the text with [hazm](https://github.com/roshan-research/hazm);
2. builds a TF-IDF vector space over the corpus (implemented manually, without scikit-learn);
3. ranks documents by cosine similarity to the query;
4. evaluates the ranking on the 50 benchmark queries and prints sample searches.

## Features

- Persian preprocessing: hazm `Normalizer`, lowercasing, removal of non-Arabic-script characters, hazm `word_tokenize`, stop-word removal, hazm `Stemmer`
- Manual TF-IDF: raw term frequency × `ln(N / df)`, L2-normalized sparse vectors
- Manual cosine similarity on sparse (dictionary) vectors
- Evaluation: Precision@k, Recall@k, F1@k, Mean Average Precision (MAP)
- Command-line interface with configurable data folder and cutoff `k`
- 50 built-in sample searches (the benchmark queries) with ranked results

## Methodology

| Step | Implementation |
|---|---|
| Document id | `<year-folder>_<file name>`, e.g. `2004_HAM2-830506-029.ham` |
| Term weight | `tf × idf`, with `tf` = raw count and `idf = ln(N / df)`; terms with zero weight are dropped |
| Normalization | Each document and query vector is L2-normalized |
| Similarity | Dot product of normalized vectors (= cosine similarity); only documents with similarity > 0 are returned, up to 1000 |
| Query vector | Built with the same vocabulary and IDF values as the documents |
| Metrics | P@k = hits / k; R@k = hits / number of relevant docs; F1 from P and R; AP = sum of precision at each relevant rank / number of relevant docs; MAP = mean AP over evaluated queries |

## Results

Evaluated on **50 queries**, **k = 10** (output of the original run is in [`results/sample_output.txt`](results/sample_output.txt)):

| Metric | Value |
|---|---|
| MAP | 0.7423 |
| Mean Precision@10 | 0.7680 |
| Mean Recall@10 | 0.3219 |
| Mean F1@10 | 0.3783 |

## Project structure

```
persian-tfidf-retrieval-hamshahri/
├── README.md
├── .gitignore
├── src/
│   └── ir_system.py          # preprocessing, TF-IDF, retrieval, evaluation, CLI
├── data/
│   └── Hamshahri            
├── examples/
│   └── tfidf_worked_example.ipynb   # small TF-IDF weighting exercise (ltc-style table)
└── results/
    └── sample_output.txt     # evaluation + sample searches from the original run
```

## Worked example

[`examples/tfidf_worked_example.ipynb`](examples/tfidf_worked_example.ipynb) is a small standalone exercise that prints a step-by-step TF-IDF table for one query and one document: log-scaled term frequency (`1 + log10 tf`), `idf = log10(N / df)`, L2 normalization and the resulting cosine similarity (0.828 for the sample in the notebook). It needs `pandas` (`pip install pandas`) and is independent of the retrieval system above, which uses raw term frequency and natural log.

## Prerequisites

- Python 3.11 (the version used by the author; other 3.x versions are untested)
- The Hamshahri corpus, queries, relevance judgements and a Persian stop-word list (see [`data/README.md`](data/README.md)). The data is **not** included in this repository.

## Installation

```bash
git clone https://github.com/<username>/persian-tfidf-retrieval-hamshahri.git
cd persian-tfidf-retrieval-hamshahri
python -m venv .venv
.venv\Scripts\activate        # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

## Usage

Place the dataset in `data/HamshahriData/` (see below), then run:

```bash
python src/ir_system.py
```

Options:

```bash
python src/ir_system.py --data-dir path/to/HamshahriData   # custom data folder
python src/ir_system.py --k 20                             # evaluate at k = 20
python src/ir_system.py --no-demo                          # evaluation only, skip sample searches
```

Indexing is done in pure Python and may take a while on the full corpus. The program prints the evaluation first and then the sample searches.

## Dataset

Expected layout (names are the ones used by the code):

```
HamshahriData/
├── HamshahriCorpus/<year>/*.ham
├── Queries/<id>.q
├── RelativeAssesemnt/judgements.txt     # one "<query id> <document id>" pair per line
└── persian_stopwords.txt                # one stop word per line
```

The Hamshahri corpus is a third-party dataset and is not redistributed here. Obtain it from its official source and follow its terms of use.

## Limitations

- Retrieval compares the query against every document vector (no inverted index), so queries are slow on large corpora.
- The whole corpus is held in memory.
- Plain TF-IDF with raw term frequency; no query expansion, no BM25, no comparison against other models in this repository.
- Evaluation depends on the provided relevance judgements, whose document ids must match the `<file name>` part of the document ids.

## Future improvements

- Build an inverted index to speed up retrieval
- Add log-scaled TF and compare with BM25
- Add unit tests for TF-IDF, cosine similarity and the metrics
- Cache the built index to disk

## Acknowledgements and license

- Text processing: [hazm](https://github.com/roshan-research/hazm)
- Corpus: Hamshahri corpus (third-party; see its license)
- Code license: `<choose a license, e.g. MIT>`
