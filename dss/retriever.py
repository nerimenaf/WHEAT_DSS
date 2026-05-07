from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class Passage:
    passage_id: str
    source: str
    text: str
    score: float


def chunk_paragraphs(text: str) -> list[str]:
    # Split on blank lines; keep reasonably sized chunks
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paras


class TfidfParagraphRetriever:
    def __init__(self, kb_files: list[Path]):
        self.kb_files = kb_files
        self.passages: list[tuple[str, str, str]] = []  # (passage_id, source, text)

        for f in kb_files:
            raw = f.read_text(encoding="utf-8")
            paras = chunk_paragraphs(raw)
            for i, p in enumerate(paras):
                pid = f"{f.name}::p{i+1}"
                self.passages.append((pid, f.name, p))

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform([p[2] for p in self.passages])

    def retrieve(self, query: str, top_k: int = 3) -> list[Passage]:
        qv = self.vectorizer.transform([query])
        scores = (self.matrix @ qv.T).toarray().ravel()
        if scores.size == 0:
            return []

        idxs = np.argsort(scores)[::-1][:top_k]
        results: list[Passage] = []
        for idx in idxs:
            pid, src, txt = self.passages[int(idx)]
            results.append(Passage(passage_id=pid, source=src, text=txt, score=float(scores[int(idx)])))
        return results


def format_passages(passages: list[Passage]) -> str:
    lines = []
    for p in passages:
        lines.append(f"[{p.passage_id}] (score={p.score:.3f}) {p.text}")
    return "\n\n".join(lines)