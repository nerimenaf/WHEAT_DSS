from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .episodic_memory import load_episodes
from .kb_build import ChunkRecord, build_and_save_indexes  # reuse your KB index builder


def _memory_text(ep: dict[str, Any], task: str) -> tuple[str, dict[str, Any]]:
    ctx = ep.get("context_t", {}) or {}
    a = ctx.get("analytics", {}) or {}
    w = ctx.get("weather", {}) or {}
    f = ctx.get("field", {}) or {}
    stage = a.get("phenological_stage", "General")
    crop = f.get("crop_type", "wheat")

    dec = (ep.get("decisions_t", {}) or {}).get(task, {}) or {}
    out = ep.get("outcome_t", {}) or {}
    fb = float((ep.get("feedback_t", {}) or {}).get("score", 0.0) or 0.0)

    text = (
        f"EPISODE {ep.get('episode_id')}\n"
        f"Task={task}; Crop={crop}; Stage={stage}\n"
        f"SoilMoisture={f.get('soil_moisture_pct')}%; N/P/K={f.get('n_ppm')}/{f.get('p_ppm')}/{f.get('k_ppm')} ppm\n"
        f"Rain24h={w.get('rain_next_24h_mm')}mm\n"
        f"Decision={dec.get('decision')}; Timing={dec.get('timing')}; Quantity={dec.get('quantity','')}\n"
        f"Outcome: dMoist={out.get('delta_soil_moisture_pct')}, dN={out.get('delta_n_ppm')}, dP={out.get('delta_p_ppm')}, dK={out.get('delta_k_ppm')}\n"
        f"Feedback={fb}\n"
    )

    meta = {
        "task": task,
        "crop_type": crop,
        "stage_tags": [stage, "General"],
        "feedback_score": fb,
        "source": "local_memory",
    }
    return text, meta


def rebuild_local_memory_indexes(
    *,
    threshold: float = 0.75,
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> None:
    out_root = Path("memory/index")
    out_root.mkdir(parents=True, exist_ok=True)

    episodes = load_episodes()

    for task in ["irrigation", "fertilization"]:
        records: list[ChunkRecord] = []

        for ep in episodes:
            fb = float((ep.get("feedback_t", {}) or {}).get("score", 0.0) or 0.0)
            if fb < threshold:
                continue

            outcome = ep.get("outcome_t")
            if not outcome:
                continue

            latest = (outcome.get("latest_actions", {}) or {}).get(task, {})
            if latest.get("status") != "done":
                continue

            text, meta = _memory_text(ep, task)

            records.append(ChunkRecord(
                chunk_id=f"memory::{task}::{ep.get('episode_id')}",
                source_file="local_memory",
                text=text,
                metadata=meta,
            ))

        out_dir = out_root / task

        # ---- IMPORTANT: if no records, remove old index files and skip building ----
        if not records:
            out_dir.mkdir(parents=True, exist_ok=True)
            for fname in ["dense.faiss", "bm25.json", "chunks.jsonl", "meta.json"]:
                p = out_dir / fname
                if p.exists():
                    p.unlink()
            continue

        build_and_save_indexes(records, out_dir, embed_model_name=embed_model_name)