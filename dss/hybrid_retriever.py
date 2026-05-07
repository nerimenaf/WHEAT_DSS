from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import faiss  # type: ignore
from rank_bm25 import BM25Okapi  # type: ignore
from sentence_transformers import SentenceTransformer, CrossEncoder  # type: ignore


@dataclass
class Passage:
    passage_id: str
    source: str
    text: str
    score: float
    metadata: dict[str, Any]


def tokenize_for_bm25(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def stage_from_context(ctx: dict[str, Any]) -> str:
    return (ctx.get("analytics", {}) or {}).get("phenological_stage", "General")


class HybridRAGStore:
    """
    Loads:
    - chunk records (jsonl)
    - dense FAISS index
    - BM25 sparse index
    - embedding model
    """
    def __init__(
        self,
        store_dir: Path,
        *,
        embed_model_name: str | None = None,
        cross_encoder_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        use_reranker: bool = True,
    ):
        self.store_dir = store_dir
        self.chunks = self._load_chunks(store_dir / "chunks.jsonl")
        self.id_to_pos = {c["chunk_id"]: i for i, c in enumerate(self.chunks)}

        # Dense
        self.faiss_index = faiss.read_index(str(store_dir / "dense.faiss"))
        meta = json.loads((store_dir / "meta.json").read_text(encoding="utf-8"))
        self.embed_model_name = embed_model_name or meta["embedding_model"]
        self.embedder = SentenceTransformer(self.embed_model_name)

        # Sparse BM25
        bm25_payload = json.loads((store_dir / "bm25.json").read_text(encoding="utf-8"))
        self.bm25_chunk_ids = bm25_payload["chunk_ids"]
        self.bm25_tokens = bm25_payload["tokenized_corpus"]
        self.bm25 = BM25Okapi(self.bm25_tokens)

        # Reranker
        self.use_reranker = use_reranker
        self.reranker = CrossEncoder(cross_encoder_name) if use_reranker else None

    def _load_chunks(self, path: Path) -> list[dict[str, Any]]:
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows


class HybridRetriever:
    """
    Multi-step retrieval:
      1) Candidate retrieval: dense + BM25 (hybrid)
      2) Metadata filtering: crop_type, task, stage
      3) Re-ranking: cross-encoder
    """
    def __init__(
        self,
        store: HybridRAGStore,
        *,
        alpha: float = 0.6,      # weight for dense vs sparse
        dense_k: int = 20,
        sparse_k: int = 20,
        final_k: int = 5,
    ):
        self.store = store
        self.alpha = alpha
        self.dense_k = dense_k
        self.sparse_k = sparse_k
        self.final_k = final_k

    def _metadata_filter(self, ctx: dict[str, Any], chunk: dict[str, Any]) -> bool:
        crop_type = (ctx.get("field", {}) or {}).get("crop_type", "wheat")
        stage = stage_from_context(ctx)
        task = (ctx.get("agent_task") or "")  # passed in by agent

        md = chunk.get("metadata", {}) or {}
        if md.get("crop_type") != crop_type:
            return False
        if task and md.get("task") != task:
            return False

        # stage filtering: allow "General" chunks or exact tag match
        tags = md.get("stage_tags", ["General"])
        return ("General" in tags) or (stage in tags)

    def retrieve(self, query: str, ctx: dict[str, Any]) -> list[Passage]:
        # --- Dense candidates ---
        q_emb = self.store.embedder.encode([query], normalize_embeddings=True)
        q_emb = np.asarray(q_emb, dtype="float32")

        dense_scores, dense_idx = self.store.faiss_index.search(q_emb, self.dense_k)
        dense_scores = dense_scores.ravel().tolist()
        dense_idx = dense_idx.ravel().tolist()

        dense_candidates: dict[str, float] = {}
        for score, idx in zip(dense_scores, dense_idx):
            if idx < 0:
                continue
            chunk = self.store.chunks[idx]
            if self._metadata_filter(ctx, chunk):
                dense_candidates[chunk["chunk_id"]] = float(score)

        # --- Sparse candidates (BM25) ---
        q_tokens = tokenize_for_bm25(query)
        bm25_scores = self.store.bm25.get_scores(q_tokens)
        top_sparse_idx = np.argsort(bm25_scores)[::-1][: self.sparse_k]

        sparse_candidates: dict[str, float] = {}
        for i in top_sparse_idx:
            cid = self.store.bm25_chunk_ids[int(i)]
            chunk = self.store.chunks[self.store.id_to_pos[cid]]
            if self._metadata_filter(ctx, chunk):
                sparse_candidates[cid] = float(bm25_scores[int(i)])

        # --- Hybrid merge (normalize sparse into 0..1) ---
        merged_ids = set(dense_candidates) | set(sparse_candidates)
        if not merged_ids:
            return []

        sparse_vals = [sparse_candidates.get(cid, 0.0) for cid in merged_ids]
        s_min, s_max = min(sparse_vals), max(sparse_vals)
        def norm_sparse(x: float) -> float:
            return 0.0 if s_max == s_min else (x - s_min) / (s_max - s_min)

        hybrid: list[tuple[str, float]] = []
        for cid in merged_ids:
            d = dense_candidates.get(cid, 0.0)  # already cosine-ish
            s = norm_sparse(sparse_candidates.get(cid, 0.0))
            score = self.alpha * d + (1.0 - self.alpha) * s
            hybrid.append((cid, float(score)))

        # take a candidate pool before reranking
        hybrid.sort(key=lambda x: x[1], reverse=True)
        pool = hybrid[: max(self.final_k * 4, 20)]

        # --- Re-ranking with cross-encoder ---
        if self.store.use_reranker and self.store.reranker is not None:
            pairs = []
            pool_chunks = []
            for cid, _ in pool:
                ch = self.store.chunks[self.store.id_to_pos[cid]]
                pairs.append((query, ch["text"]))
                pool_chunks.append(ch)

            rerank_scores = self.store.reranker.predict(pairs).tolist()
            reranked = list(zip(pool_chunks, rerank_scores))
            reranked.sort(key=lambda x: x[1], reverse=True)
            chosen = reranked[: self.final_k]

            out: list[Passage] = []
            for ch, sc in chosen:
                out.append(Passage(
                    passage_id=ch["chunk_id"],
                    source=ch["source_file"],
                    text=ch["text"],
                    score=float(sc),
                    metadata=ch.get("metadata", {}) or {},
                ))
            return out

        # If no reranker, return top final_k from hybrid
        out: list[Passage] = []
        for cid, sc in hybrid[: self.final_k]:
            ch = self.store.chunks[self.store.id_to_pos[cid]]
            out.append(Passage(
                passage_id=ch["chunk_id"],
                source=ch["source_file"],
                text=ch["text"],
                score=float(sc),
                metadata=ch.get("metadata", {}) or {},
            ))
        return out


def format_passages(passages: list[Passage]) -> str:
    lines = []
    for p in passages:
        lines.append(f"[{p.passage_id}] score={p.score:.3f} | meta={p.metadata}\n{p.text}")
    return "\n\n".join(lines)