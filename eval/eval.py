#!/usr/bin/env python3
"""eval.py — Retrieval evaluation: hit@k, MRR and top-score signal.

Usage:
    python eval.py --retriever v0 --k 3,5,10 --results results/v0.json

Frozen contract with team B:
    retrieve(query: str, k: int) -> list[dict]
    each dict: {"text": str, "source": str, "page": int, "score": float}
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import unicodedata
from pathlib import Path

from adapters import get_retrieve

TEXT_LIMIT = 200


# ---------------------------------------------------------------------------
# 1. Test set — questions.jsonl, one question per line:
# {
#   "id": "q001",
#   "question": "¿…?",
#   "answerable": true,
#   "expected": [{"source": "gaceta_2024.pdf", "pages": [12, 13]}],
#   "keywords": ["distinctive phrase"]
# }
# Out-of-corpus questions: "answerable": false, no "expected".
# Lines starting with "//" are ignored (comments).
# ---------------------------------------------------------------------------

def load_questions(path: str) -> list[dict]:
    questions = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith("//"):
                questions.append(json.loads(line))
    if not questions:
        sys.exit(f"Empty test set: {path}")
    return questions


def _normalize(text: str) -> str:
    """Lowercase, strip diacritics (á -> a, ñ -> n) and collapse whitespace so
    Spanish text and keywords match regardless of accents or stray spaces."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    without_diacritics = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(without_diacritics.split())


def is_relevant(hit: dict, question: dict) -> bool:
    """Does the retrieved chunk count as relevant for this question?

    Two strategies (either one is enough):
      1. source + page match against "expected" (exact filename comparison)
      2. all "keywords" appear in the chunk text (case/accent-insensitive,
         tolerant to PDF parsing issues)
    """
    for exp in question.get("expected", []):
        if hit.get("source") == exp.get("source"):
            pages = exp.get("pages")
            if not pages or hit.get("page") in pages:
                return True
    keywords = question.get("keywords", [])
    if keywords:
        text = _normalize(hit.get("text") or "")
        if all(_normalize(keyword) in text for keyword in keywords):
            return True
    return False


# ---------------------------------------------------------------------------
# 2. Metrics
# ---------------------------------------------------------------------------

def evaluate_question(question: dict, retrieve, ks: list[int]) -> dict:
    """Evaluate one question: search once with max(k), compute all metrics."""
    answerable = question.get("answerable", True)
    hits = retrieve(question["question"], max(ks))

    result = {
        "id": question["id"],
        "answerable": answerable,
        "top_score": hits[0].get("score", 0.0) if hits else 0.0,
        "first_relevant_rank": None,
        "n_relevant_in_topk": {k: 0 for k in ks},
        "hits": [],
    }

    for rank, hit in enumerate(hits, start=1):
        relevant = answerable and is_relevant(hit, question)
        result["hits"].append(
            {
                "rank": rank,
                "source": hit.get("source"),
                "page": hit.get("page"),
                "score": hit.get("score"),
                "relevant": relevant,
                "text": (hit.get("text") or "")[:TEXT_LIMIT],
            }
        )
        if not relevant:
            continue
        if result["first_relevant_rank"] is None:
            result["first_relevant_rank"] = rank
        for k in ks:
            if rank <= k:
                result["n_relevant_in_topk"][k] += 1
    return result


def summarize(results: list[dict], ks: list[int]) -> dict:
    """hit@k and MRR over answerable questions; score stats for the rest."""
    answerable = [r for r in results if r["answerable"]]
    unanswerable = [r for r in results if not r["answerable"]]

    summary = {}
    for k in ks:
        hits = sum(1 for r in answerable if r["n_relevant_in_topk"][k] > 0)
        summary[f"hit@{k}"] = round(hits / len(answerable), 4) if answerable else None

    # Standard MRR over ALL answerable questions: a question with no relevant
    # hit contributes 0, so a retriever that finds nothing scores 0.
    rr = [1.0 / r["first_relevant_rank"] if r["first_relevant_rank"] else 0.0 for r in answerable]
    summary["mrr"] = round(statistics.mean(rr), 4) if rr else 0.0
    summary["questions_answerable"] = len(answerable)
    summary["questions_hors_corpus"] = len(unanswerable)

    if unanswerable:
        scores = [r["top_score"] for r in unanswerable]
        summary["hors_corpus_top_score"] = {
            "mean": round(statistics.mean(scores), 4),
            "max": round(max(scores), 4),
        }
    return summary


# ---------------------------------------------------------------------------
# 3. Execution
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval evaluation: hit@k and MRR for the RAG")
    parser.add_argument("--questions", default="questions.jsonl")
    parser.add_argument("--retriever", required=True, help="v0 | v1 | v2 | v3 | stub | ...")
    parser.add_argument("--k", default="3,5,10", help="comma-separated list of k values")
    parser.add_argument("--results", default=None, help="output JSON (per question, raw hits)")
    parser.add_argument("--summary", default=None, help="output JSON (aggregated)")
    args = parser.parse_args()

    ks = sorted({int(k) for k in args.k.split(",")})
    retrieve = get_retrieve(args.retriever)
    questions = load_questions(args.questions)

    print(f"Evaluating {len(questions)} questions with retriever '{args.retriever}' (k={ks})...")
    results = [evaluate_question(question, retrieve, ks) for question in questions]
    summary = summarize(results, ks)

    print("\n=== Summary ===")
    for metric, value in summary.items():
        if isinstance(value, dict):
            print(f"  {metric}: " + ", ".join(f"{key}={val}" for key, val in value.items()))
        else:
            print(f"  {metric}: {value}")

    misses = [r["id"] for r in results if r["answerable"] and r["first_relevant_rank"] is None]
    if misses:
        print(f"\nAnswerable questions with no relevant hit: {', '.join(misses)}")

    # JSON outputs (always save the raw run: metrics can be recomputed later)
    if args.results:
        Path(args.results).parent.mkdir(parents=True, exist_ok=True)
        Path(args.results).write_text(
            json.dumps({"summary": summary, "results": results}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nDetailed results -> {args.results}")
    if args.summary:
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Summary -> {args.summary}")


if __name__ == "__main__":
    main()
