#!/usr/bin/env python3
"""adapters.py — Bridge between the evaluation harness and each retriever version.

Frozen contract with team B:
    retrieve(query: str, k: int) -> list[dict]
    each dict: {"text": str, "source": str, "page": int, "score": float}

Version map:
    v0   -> retrieval_tfidf.retrieve    (B, TF-IDF)
    v1   -> retrieval_dense.retrieve    (A, dense)
    v2   -> retrieval_hybrid.retrieve   (C, hybrid)
    v3   -> retrieval_hybrid.retrieve   (C, hybrid, next iteration)
    stub -> built-in fake retriever for harness self-tests
"""

from __future__ import annotations

import sys
import unicodedata

RETRIEVER_MODULES = {
    "v0": "retrieval_tfidf",
    "v1": "retrieval_dense",
    "v2": "retrieval_hybrid",
    "v3": "retrieval_hybrid",
}


def get_retrieve(retriever_name: str):
    """Return the retrieve(query, k) function for the requested retriever."""
    if retriever_name == "stub":
        return stub_retrieve

    try:
        module_name = RETRIEVER_MODULES[retriever_name]
    except KeyError:
        known = ", ".join(sorted(RETRIEVER_MODULES)) + ", stub"
        sys.exit(f"Unknown retriever: '{retriever_name}'. Known retrievers: {known}")

    try:
        module = __import__(module_name)
    except ImportError as exc:
        sys.exit(
            f"Could not import '{module_name}': {exc}\n"
            "Ask your teammate for the module, or run with --retriever stub for a self-test."
        )
    return getattr(module, "retrieve")


# ---------------------------------------------------------------------------
# Stub retriever — self-test only, no dependency on the real retrievers.
# Built-in Spanish sample chunks plus naive token-overlap scoring, so the full
# pipeline (search -> ranking -> relevance check -> metrics) can be validated
# before any real retriever is delivered.
# ---------------------------------------------------------------------------

_STUB_CHUNKS = [
    {
        "text": (
            "El organismo aprobó el presupuesto extraordinario de 2024 por un total "
            "de 45 millones de euros, destinados principalmente a infraestructuras "
            "y digitalización."
        ),
        "source": "gaceta_2024.pdf",
        "page": 12,
    },
    {
        "text": (
            "El presupuesto extraordinario de 2024 incluye una partida adicional de "
            "8 millones de euros para programas de formación interna."
        ),
        "source": "gaceta_2024.pdf",
        "page": 13,
    },
    {
        "text": (
            "La sesión ordinaria del organismo revisó el acta anterior y aprobó el "
            "calendario de reuniones para el primer trimestre."
        ),
        "source": "gaceta_2024.pdf",
        "page": 11,
    },
    {
        "text": (
            "Se publicaron las bases reguladoras de las subvenciones para proyectos "
            "de investigación aplicada."
        ),
        "source": "gaceta_2024.pdf",
        "page": 25,
    },
    {
        "text": (
            "El informe anual de 2023 resume las actividades de la organización "
            "internacional durante el ejercicio."
        ),
        "source": "informe_anual_2023.pdf",
        "page": 3,
    },
    {
        "text": (
            "Los ingresos totales ascendieron a 120 millones de euros, un 6% más que "
            "el año anterior."
        ),
        "source": "informe_anual_2023.pdf",
        "page": 7,
    },
    {
        "text": (
            "El capítulo de recursos humanos describe la plantilla media y los planes "
            "de contratación."
        ),
        "source": "informe_anual_2023.pdf",
        "page": 18,
    },
    {
        "text": (
            "Las conclusiones destacan la mejora de la eficiencia administrativa y la "
            "reducción de plazos de tramitación."
        ),
        "source": "informe_anual_2023.pdf",
        "page": 42,
    },
]


def _normalize(text: str) -> str:
    """Lowercase and strip diacritics (á -> a, ñ -> n) so Spanish terms match
    regardless of accents."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _tokens(text: str) -> set:
    normalized = _normalize(text)
    cleaned = "".join(char if char.isalnum() else " " for char in normalized)
    return {token for token in cleaned.split() if len(token) >= 4}


def stub_retrieve(query: str, k: int) -> list:
    """Naive token-overlap retriever over the built-in sample chunks.

    Ties keep the order of _STUB_CHUNKS (Python's sort is stable), so results
    are fully deterministic.
    """
    query_tokens = _tokens(query)
    scored = []
    for chunk in _STUB_CHUNKS:
        overlap = len(query_tokens & _tokens(chunk["text"]))
        score = overlap / len(query_tokens) if query_tokens else 0.0
        scored.append({**chunk, "score": round(score, 4)})
    scored.sort(key=lambda hit: hit["score"], reverse=True)
    return scored[:k]
