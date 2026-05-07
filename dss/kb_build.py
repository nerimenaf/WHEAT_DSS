from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import faiss  # type: ignore
from rank_bm25 import BM25Okapi  # type: ignore
import tiktoken  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore


STAGE_KEYWORDS = {
    "Germination / Emergence": ["germination", "emergence", "seedling"],
    "Tillering": ["tillering", "tiller"],
    "Stem elongation": ["stem elongation", "jointing", "booting"],
    "Heading / Flowering": ["heading", "flowering", "anthesis"],
    "Grain filling": ["grain filling", "milk", "dough"],
    "Maturity / Late season": ["maturity", "ripening", "harvest"],
}


def detect_stage_tags(text: str) -> list[str]:
    t = text.lower()
    tags = []
    for stage, kws in STAGE_KEYWORDS.items():
        if any(kw in t for kw in kws):
            tags.append(stage)
    return tags or ["General"]


def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def split_to_semantic_blocks(raw: str) -> list[str]:
    """
    A simple semantic split: paragraphs separated by blank lines.
    """
    blocks = [b.strip() for b in raw.split("\n\n") if b.strip()]
    return blocks


def chunk_by_tokens(
    blocks: list[str],
    *,
    min_tokens: int = 200,
    max_tokens: int = 450,
    overlap_tokens: int = 50,
    encoding_name: str = "cl100k_base",
) -> list[str]:
    """
    Build chunks ~200–450 tokens with overlap to preserve continuity.
    Uses tiktoken token counting.
    """
    enc = tiktoken.get_encoding(encoding_name)

    chunks: list[str] = []
    cur: list[str] = []
    cur_tokens = 0

    def tokens_count(s: str) -> int:
        return len(enc.encode(s))

    for b in blocks:
        b = normalize_whitespace(b)
        bt = tokens_count(b)

        # if a single block is too big, hard-split it by sentences
        if bt > max_tokens:
            sentences = re.split(r"(?<=[.!?])\s+", b)
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                st = tokens_count(sent)
                if cur_tokens + st > max_tokens and cur_tokens >= min_tokens:
                    chunks.append(" ".join(cur).strip())
                    # overlap
                    if overlap_tokens > 0:
                        chunk_tokens = enc.encode(chunks[-1])
                        overlap = enc.decode(chunk_tokens[-overlap_tokens:])
                        cur = [overlap]
                        cur_tokens = tokens_count(overlap)
                    else:
                        cur, cur_tokens = [], 0
                cur.append(sent)
                cur_tokens += st
            continue

        # normal accumulation
        if cur_tokens + bt > max_tokens and cur_tokens >= min_tokens:
            chunks.append(" ".join(cur).strip())
            if overlap_tokens > 0:
                chunk_tokens = enc.encode(chunks[-1])
                overlap = enc.decode(chunk_tokens[-overlap_tokens:])
                cur = [overlap]
                cur_tokens = tokens_count(overlap)
            else:
                cur, cur_tokens = [], 0

        cur.append(b)
        cur_tokens += bt

    if cur:
        chunks.append(" ".join(cur).strip())

    return [c for c in chunks if c]


@dataclass
class ChunkRecord:
    chunk_id: str
    source_file: str
    text: str
    metadata: dict[str, Any]


def build_chunk_records(kb_file: Path, task: str, crop_type: str = "wheat") -> list[ChunkRecord]:
    raw = read_text(kb_file)
    blocks = split_to_semantic_blocks(raw)
    chunks = chunk_by_tokens(blocks)

    records: list[ChunkRecord] = []
    for i, ch in enumerate(chunks, start=1):
        rec = ChunkRecord(
            chunk_id=f"{kb_file.name}::c{i}",
            source_file=kb_file.name,
            text=ch,
            metadata={
                "task": task,                 # irrigation / fertilization
                "crop_type": crop_type,        # wheat
                "stage_tags": detect_stage_tags(ch),  # list of stage tags
            },
        )
        records.append(rec)
    return records


def save_jsonl(records: list[ChunkRecord], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({
                "chunk_id": r.chunk_id,
                "source_file": r.source_file,
                "text": r.text,
                "metadata": r.metadata,
            }, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def tokenize_for_bm25(text: str) -> list[str]:
    # simple tokenization
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def build_and_save_indexes(
    records: list[ChunkRecord],
    out_dir: Path,
    *,
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Save JSONL chunks
    jsonl_path = out_dir / "chunks.jsonl"
    save_jsonl(records, jsonl_path)

    # 2) Dense embeddings + FAISS (cosine similarity via normalized inner product)
    model = SentenceTransformer(embed_model_name)
    texts = [r.text for r in records]
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    emb = np.asarray(emb, dtype="float32")

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, str(out_dir / "dense.faiss"))

    # 3) Sparse BM25
    tokenized = [tokenize_for_bm25(t) for t in texts]
    bm25 = BM25Okapi(tokenized)

    # Save BM25 corpus tokens and id mapping
    bm25_path = out_dir / "bm25.json"
    bm25_payload = {
        "tokenized_corpus": tokenized,
        "chunk_ids": [r.chunk_id for r in records],
    }
    bm25_path.write_text(json.dumps(bm25_payload), encoding="utf-8")

    # Save embedding model name
    (out_dir / "meta.json").write_text(json.dumps({
        "embedding_model": embed_model_name,
        "num_chunks": len(records),
    }, indent=2), encoding="utf-8")


def main():
    kb_irrig = Path("knowledge/irrigation.txt")
    kb_fert = Path("knowledge/fertilization.txt")
    out_root = Path("knowledge/index")

    # Build irrigation KB
    irrig_records = build_chunk_records(kb_irrig, task="irrigation", crop_type="wheat")
    build_and_save_indexes(irrig_records, out_root / "irrigation")

    # Build fertilization KB
    fert_records = build_chunk_records(kb_fert, task="fertilization", crop_type="wheat")
    build_and_save_indexes(fert_records, out_root / "fertilization")

    print("KB build done.")
    print(f"- {out_root / 'irrigation'}")
    print(f"- {out_root / 'fertilization'}")


if __name__ == "__main__":
    main()