# Gaceta RAG Chatbot

A **RAG** (Retrieval-Augmented Generation) chatbot that answers questions about documents from the Gaceta Oficial (Spanish-language corpus).

## Project status

- [x] Corpus ingestion — 49 Gaceta documents converted to Markdown (`data/markdown/`, ~16 MB, tracked in git); chunking lands with the retrieval pipeline
- [x] Offline evaluation harness (`eval/`) — hit@k, MRR, out-of-corpus scoring, version comparison workflow
- [x] Embeddings and vector index (chromadb)
- [x] Retrieval pipeline — V0 TF-IDF · V1 dense · V2 Multi-Query · V3 HyDe
- [ ] Generation layer, chat API and interface

## Layout

```
gaceta-rag-chatbot/
├── data/
│   ├── markdown/     # Gaceta corpus in Markdown (tracked in git)
│   └── raw/          # Original PDFs and corpus_colab.zip (gitignored)
├── eval/             # Offline retrieval evaluation harness — see eval/README.md
├── rag.ipynb         # Main notebook 
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.11+
- Install dependencies: `pip install -r requirements.txt` (scikit-learn, chromadb, plotly)

## Corpus

- `data/markdown/`: 49 Gaceta documents converted to Markdown (~16 MB), versioned in this repository.
- `data/raw/`: original PDFs and `corpus_colab.zip` (~430 MB), available locally but outside version control.

## Evaluation

The retrieval layer is measured offline in `eval/` against a 25-question test set (20 answerable, 5 out-of-corpus). From `eval/`:

```
# harness self-test (no retriever modules needed)
python eval.py --retriever stub --questions stub_questions.jsonl --k 3,5,10 --results results/stub.json

# evaluate a real retriever version
python eval.py --retriever v0 --k 3,5,10 --results results/v0.json
```

Metrics: **hit@k** (at least one relevant chunk in the top-k), **MRR**, and a **top-score signal** for out-of-corpus questions (calibrates the "do not answer" threshold). Runs are archived in `eval/results/` and compared in `eval/reports/comparison.md`. Full documentation — retriever contract, question-set schema, metric definitions — in [`eval/README.md`](eval/README.md).
