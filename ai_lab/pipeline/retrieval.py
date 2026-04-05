from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

from ai_lab.pipeline.chunking import Chunk


TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float
    matched_terms: tuple[str, ...]


class SparseVectorIndex:
    def __init__(self, chunks: list[Chunk], idf: dict[str, float], vectors: list[dict[str, float]]) -> None:
        self.chunks = chunks
        self.idf = idf
        self.vectors = vectors

    @classmethod
    def build(cls, chunks: Iterable[Chunk]) -> "SparseVectorIndex":
        chunk_list = list(chunks)
        document_frequency: Counter[str] = Counter()
        tokenized_chunks: list[Counter[str]] = []

        for chunk in chunk_list:
            tokens = Counter(_tokenize(chunk.text))
            tokenized_chunks.append(tokens)
            document_frequency.update(tokens.keys())

        total_docs = max(len(chunk_list), 1)
        idf = {
            token: math.log((1 + total_docs) / (1 + freq)) + 1.0
            for token, freq in document_frequency.items()
        }

        vectors = [_normalize(_tfidf_vector(tokens, idf)) for tokens in tokenized_chunks]
        return cls(chunk_list, idf, vectors)

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        query_tokens = Counter(_tokenize(query))
        query_vector = _normalize(_tfidf_vector(query_tokens, self.idf))
        if not query_vector:
            return []

        hits: list[SearchHit] = []
        query_terms = tuple(query_tokens.keys())
        for chunk, vector in zip(self.chunks, self.vectors):
            score = _dot(query_vector, vector)
            if score <= 0:
                continue
            matched = tuple(term for term in query_terms if term in vector)
            hits.append(SearchHit(chunk=chunk, score=score, matched_terms=matched))

        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]

    def to_payload(self) -> dict[str, Any]:
        return {
            "idf": self.idf,
            "vectors": self.vectors,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any], chunks: list[Chunk]) -> "SparseVectorIndex":
        return cls(
            chunks=chunks,
            idf={str(key): float(value) for key, value in payload.get("idf", {}).items()},
            vectors=[
                {str(token): float(weight) for token, weight in vector.items()}
                for vector in payload.get("vectors", [])
            ],
        )


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _tfidf_vector(tokens: Counter[str], idf: dict[str, float]) -> dict[str, float]:
    if not tokens:
        return {}
    max_tf = max(tokens.values())
    return {
        token: (count / max_tf) * idf.get(token, 1.0)
        for token, count in tokens.items()
    }


def _normalize(vector: dict[str, float]) -> dict[str, float]:
    magnitude = math.sqrt(sum(weight * weight for weight in vector.values()))
    if not magnitude:
        return {}
    return {token: weight / magnitude for token, weight in vector.items()}


def _dot(left: dict[str, float], right: dict[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(weight * right.get(token, 0.0) for token, weight in left.items())
