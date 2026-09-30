# RAG Retrieval Evaluation

Offline evaluation harness for the retrieval layer of the RAG chatbot. It measures how well a retriever places relevant passages from the source PDFs in the top-k results, against a frozen question set.

The document corpus is in Spanish; the harness itself is in English and language-agnostic (see [Spanish content](#spanish-content)).

## Layout

| Path | Purpose |
|------|---------|
| `questions.jsonl` | Test set — source of truth, frozen after J1. One JSON object per line. |
| `stub_questions.jsonl` | Single-question fixture matching the stub's sample chunks (harness self-test only). |
| `eval.py` | Evaluation entry point: hit@k, MRR, per-question report. |
| `adapters.py` | Bridge to each retriever version plus the built-in `stub`. |
| `results/` | Raw output of every run (`v0.json`, `v1.json`, ...) so metrics can be re-analyzed offline. |
| `reports/comparison.md` | V0→V4 comparison table, filled in manually after each run. |

## Quick start (self-test)

The `stub` retriever uses built-in Spanish sample chunks and needs no teammate module. Run it against the bundled fixture, not the real test set:

```
python eval.py --retriever stub --questions stub_questions.jsonl --k 3,5,10 --results results/stub.json
```

Expected result: `hit@3 = hit@5 = hit@10 = 1.0`, `mrr = 1.0`. Running the stub against the real `questions.jsonl` yields ~0 scores by design: its sample chunks do not belong to the corpus.

## Evaluating a real version

```
python eval.py --retriever v0 --k 3,5,10 --results results/v0.json
```

Options:

- `--questions PATH` — test set (default `questions.jsonl`)
- `--retriever NAME` — `v0 | v1 | v2 | v3 | stub`
- `--k LIST` — comma-separated k values (default `3,5,10`)
- `--results PATH` — per-question JSON including raw hits (recommended: always save it)
- `--summary PATH` — aggregated JSON only

After each run, copy the `summary` values into `reports/comparison.md`.

## Retriever contract (frozen interface with team B)

```python
retrieve(query: str, k: int) -> list[dict]
# each dict: {"text": str, "source": str, "page": int, "score": float}
```

- `source` must be the exact PDF filename (e.g. `gaceta_2024.pdf`).
- `page` is the 1-based page number; `score` is a float, higher = better.

Versions are wired in `adapters.py`:

| Name | Module | Owner |
|------|--------|-------|
| `v0` | `retrieval_tfidf` | B (TF-IDF) |
| `v1` | `retrieval_dense` | A (dense) |
| `v2` | `retrieval_hybrid` | C (hybrid) |
| `v3` | `retrieval_hybrid` | C (hybrid, next iteration) |

If the module is missing, the harness exits with a clear message telling you to ask your teammate. Use `--retriever stub` to keep working in the meantime.

## questions.jsonl schema

One question per line, Spanish content:

```json
{"id": "q001", "question": "¿...?", "answerable": true, "expected": [{"source": "gaceta_2024.pdf", "pages": [12, 13]}], "keywords": ["presupuesto extraordinario", "organismo"]}
```

- `answerable: true` — must have `expected` (source + pages) and ideally `keywords`.
- `answerable: false` — out-of-corpus question; omit `expected`. Only its top score is tracked (calibration signal for the "do not answer" threshold).
- Lines starting with `//` are ignored (comments).
- The test set is the source of truth and stays frozen after J1. The repository ships with a single example question as a template; replace it with your own set.

## Relevance rule

A retrieved chunk counts as relevant when **either** condition holds:

1. `source` matches and `page` is listed in `expected.pages` (exact filename comparison), or
2. **all** `keywords` appear in the chunk text as substrings (case-, accent- and whitespace-insensitive).

## Metrics

- **hit@k** — fraction of answerable questions with at least one relevant chunk in the top-k. Binary per question: for RAG generation, one good passage is enough.
- **MRR** — mean of `1 / rank of the first relevant chunk` across **all** answerable questions; a question with no relevant hit contributes 0.
- **hors_corpus_top_score** — mean/max of the top score over out-of-corpus questions; compare against the answerable top scores to calibrate a no-answer threshold.

Each `results/vX.json` stores the summary plus every question with its raw hits (`rank`, `source`, `page`, `score`, `relevant`, `text` truncated to 200 chars), so any metric can be recomputed without re-running the retrievers.

## Spanish content

- The PDFs and the questions are in Spanish; nothing in the harness assumes English.
- Keyword matching normalizes case, diacritics and whitespace runs (`organización` ≡ `organizacion`, `año` ≡ `ano`), so mismatches between extracted PDF text and `questions.jsonl` do not cause false negatives.
- `source` values are compared exactly — use the PDF filename as-is.
