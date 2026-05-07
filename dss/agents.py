from __future__ import annotations

import json
from typing import Any, Optional

from langchain_core.prompts import ChatPromptTemplate

from .hybrid_retriever import HybridRetriever, format_passages
from .schemas import AgentRecommendation
from .llm import build_llm, LlmSpec


IRRIGATION_SYSTEM = """You are the Irrigation Agent for a wheat decision support system.

You MUST use ONLY:
1) Shared Context (JSON)
2) Retrieved Knowledge Passages (KB)
3) Retrieved Memory Passages (if provided)

OUTPUT:
Return ONLY one JSON object with EXACT keys:
decision, timing, quantity, reason, citations

Rules:
- decision MUST be one of: "Irrigate", "Do not irrigate", "Delay irrigation"
- quantity MUST be a STRING with units (e.g., "25 mm" or "25 mm (≈ 250 m³ total)").
- Use the system estimate from Shared Context if available:
  - If ctx.estimates.irrigation.gross_mm == 0 => decision = "Do not irrigate" and quantity = "0 mm"
  - If ctx.estimates.irrigation.gross_mm > 0 => decision = "Irrigate" (or "Delay irrigation" if heavy rain) and quantity uses gross_mm.
- If rain_next_24h_mm >= 10 => "Delay irrigation" unless KB explicitly says emergency irrigation.
- citations MUST be a JSON array of STRINGS and MUST come from the provided passage IDs only (e.g., "irrigation.txt::c1", "memory::irrigation::..."). No empty strings.

Do NOT output keys like: context, sources, task, explanation.
"""

FERTILIZATION_SYSTEM = """You are the Fertilization Agent for a wheat decision support system.

You MUST use ONLY:
1) Shared Context (JSON)
2) Retrieved Knowledge Passages (KB)
3) Retrieved Memory Passages (if provided)

OUTPUT:
Return ONLY one JSON object with EXACT keys:
decision, timing, quantity, reason, citations

Rules:
- decision MUST be one of: "Apply fertilizer", "Hold fertilization", "Delay fertilization"
- quantity MUST be a SINGLE STRING (NOT an object). Include units (kg/ha).
- Use the system estimate from Shared Context if available:
  - If ctx.estimates.fertilization has all recommended rates = 0 => decision = "Hold fertilization" and quantity = "0 kg/ha"
  - Otherwise decision = "Apply fertilizer" (or "Delay fertilization" if heavy rain risk).
- If rain_next_24h_mm >= 10 => prefer "Delay fertilization".
- citations MUST be a JSON array of STRINGS and MUST come from the provided passage IDs only.

Do NOT output keys like: context, sources, task, explanation.
"""


def _context_to_query(ctx: dict[str, Any], agent_type: str) -> str:
    a = ctx.get("analytics", {}) or {}
    w = ctx.get("weather", {}) or {}
    f = ctx.get("field", {}) or {}

    return (
        f"Crop={f.get('crop_type','wheat')}; Agent={agent_type}; "
        f"Stage={a.get('phenological_stage')}; DAS={a.get('days_after_sowing')}; "
        f"SoilMoistureStatus={a.get('soil_moisture_status')}; SoilMoisturePct={f.get('soil_moisture_pct')}; "
        f"pHStatus={a.get('soil_ph_status')}; pH={f.get('soil_ph')}; "
        f"N_status={a.get('n_status')}; P_status={a.get('p_status')}; K_status={a.get('k_status')}; "
        f"RainNext24h_mm={w.get('rain_next_24h_mm')}; TempNext24h_C={w.get('avg_temp_next_24h_c')}; "
        f"WindMax_ms={w.get('max_wind_next_24h_ms')}"
    )


def _extract_first_json_object(text: str) -> str:
    s = (text or "").strip()
    start = s.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output.")
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start:i+1]
    raise ValueError("Unclosed JSON object in model output.")


def _parse_rec(raw_text: str) -> AgentRecommendation:
    js = _extract_first_json_object(raw_text)
    obj = json.loads(js)
    # Pydantic model has normalizer for context/sources/task etc.
    return AgentRecommendation.model_validate(obj)


def _build_llm_chain(system_text: str, llm_spec: LlmSpec):
    llm = build_llm(llm_spec)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_text),
            ("human",
             "Shared Context (JSON):\n{context_json}\n\n"
             "Retrieved Knowledge Passages (KB):\n{kb_passages}\n\n"
             "Retrieved Memory Passages:\n{memory_passages}\n\n"
             "Now produce the JSON recommendation."),
        ]
    )
    return prompt | llm


def _fallback_citations(kb_passages: list[Any], mem_passages: list[Any], max_total: int = 4) -> list[str]:
    ids: list[str] = []
    for p in kb_passages or []:
        pid = getattr(p, "passage_id", None)
        if pid:
            ids.append(str(pid))
    for p in mem_passages or []:
        pid = getattr(p, "passage_id", None)
        if pid:
            ids.append(str(pid))

    out: list[str] = []
    for x in ids:
        if x not in out:
            out.append(x)
    return out[:max_total]


class IrrigationAgent:
    def __init__(
        self,
        kb_retriever: HybridRetriever,
        llm_spec: LlmSpec,
        memory_retriever: Optional[HybridRetriever] = None,
        use_rag: bool = True,
        memory_enabled: bool = True,
    ):
        self.kb_retriever = kb_retriever
        self.memory_retriever = memory_retriever
        self.use_rag = use_rag
        self.memory_enabled = memory_enabled
        self.chain = _build_llm_chain(IRRIGATION_SYSTEM, llm_spec)

    def run(self, ctx: dict[str, Any]) -> AgentRecommendation:
        query = _context_to_query(ctx, "irrigation")
        ctx2 = dict(ctx)
        ctx2["agent_task"] = "irrigation"

        kb_passages = self.kb_retriever.retrieve(query, ctx2) if self.use_rag else []
        mem_passages = (
            self.memory_retriever.retrieve(query, ctx2)
            if (self.use_rag and self.memory_enabled and self.memory_retriever)
            else []
        )

        payload = {
            "context_json": json.dumps(ctx, indent=2, ensure_ascii=False),
            "kb_passages": format_passages(kb_passages) if kb_passages else "—",
            "memory_passages": format_passages(mem_passages) if mem_passages else "—",
        }

        msg = self.chain.invoke(payload)
        raw = getattr(msg, "content", str(msg))
        rec = _parse_rec(raw)

        if not rec.citations:
            rec.citations = _fallback_citations(kb_passages, mem_passages)

        if not rec.quantity:
            est = (ctx.get("estimates", {}) or {}).get("irrigation", {}) or {}
            gross = est.get("gross_mm", 0.0)
            total_m3 = est.get("total_m3_if_area_known", None)
            if "irrig" in (rec.decision or "").lower() and gross and gross > 0:
                rec.quantity = f"Apply ~{gross} mm" + (f" (≈ {total_m3} m³ total)" if total_m3 is not None else "")
            else:
                rec.quantity = "0 mm"

        return rec


class FertilizationAgent:
    def __init__(
        self,
        kb_retriever: HybridRetriever,
        llm_spec: LlmSpec,
        memory_retriever: Optional[HybridRetriever] = None,
        use_rag: bool = True,
        memory_enabled: bool = True,
    ):
        self.kb_retriever = kb_retriever
        self.memory_retriever = memory_retriever
        self.use_rag = use_rag
        self.memory_enabled = memory_enabled
        self.chain = _build_llm_chain(FERTILIZATION_SYSTEM, llm_spec)

    def run(self, ctx: dict[str, Any]) -> AgentRecommendation:
        query = _context_to_query(ctx, "fertilization")
        ctx2 = dict(ctx)
        ctx2["agent_task"] = "fertilization"

        kb_passages = self.kb_retriever.retrieve(query, ctx2) if self.use_rag else []
        mem_passages = (
            self.memory_retriever.retrieve(query, ctx2)
            if (self.use_rag and self.memory_enabled and self.memory_retriever)
            else []
        )

        payload = {
            "context_json": json.dumps(ctx, indent=2, ensure_ascii=False),
            "kb_passages": format_passages(kb_passages) if kb_passages else "—",
            "memory_passages": format_passages(mem_passages) if mem_passages else "—",
        }

        msg = self.chain.invoke(payload)
        raw = getattr(msg, "content", str(msg))
        rec = _parse_rec(raw)

        if not rec.citations:
            rec.citations = _fallback_citations(kb_passages, mem_passages)

        if not rec.quantity:
            est = (ctx.get("estimates", {}) or {}).get("fertilization", {}) or {}
            urea = est.get("urea_kg_ha", 0.0)
            dap = est.get("dap_kg_ha", 0.0)
            mop = est.get("mop_kg_ha", 0.0)
            if "apply" in (rec.decision or "").lower() and (urea or dap or mop):
                rec.quantity = f"Urea ~{urea} kg/ha, DAP ~{dap} kg/ha, MOP ~{mop} kg/ha."
            else:
                rec.quantity = "0 kg/ha"

        return rec