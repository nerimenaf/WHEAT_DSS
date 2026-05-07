from __future__ import annotations
import json
from pydantic import BaseModel, Field, ConfigDict, model_validator
from typing import List, Dict, Any


class AgentRecommendation(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    decision: str = Field(...)
    timing: str = Field(...)

    # keep as str for consistency, but we will coerce numbers -> str
    quantity: str = Field("", description="Amount/rate with units")

    # allow the model to output "context" instead of "reason"
    reason: str = Field(..., alias="context")

    # allow the model to output "sources" instead of "citations"
    citations: List[str] = Field(default_factory=list, alias="sources")

    @model_validator(mode="before")
    @classmethod
    def normalize_llm_output(cls, data: Any):
        if not isinstance(data, dict):
            return data

        # Map alternate keys
        if "reason" not in data and "context" in data:
            data["reason"] = data["context"]
        if "citations" not in data and "sources" in data:
            data["citations"] = data["sources"]

        # If model outputs quantity as a number, convert to string
        q = data.get("quantity", "")
        if isinstance(q, (int, float)):
            data["quantity"] = str(q)

        # If model outputs quantity as an object, convert to a compact string
        if isinstance(q, dict):
            data["quantity"] = json.dumps(q, ensure_ascii=False)

        # Allow "task" to serve as reason if model messed up
        if "reason" not in data and "task" in data and isinstance(data["task"], str):
            data["reason"] = data["task"]

        # Normalize citations to List[str]
        cits = data.get("citations", [])
        if isinstance(cits, list):
            out: list[str] = []
            for c in cits:
                if isinstance(c, str):
                    out.append(c)
                elif isinstance(c, dict):
                    src = c.get("source", "")
                    pid = c.get("passage_id", "")
                    if pid and src:
                        out.append(pid if "::" in pid else f"{src}::{pid}")
                    elif pid:
                        out.append(str(pid))
            data["citations"] = out

        return data


class OrchestratedOutput(BaseModel):
    priority_order: List[str]
    combined_notes: str
    weekly_plan: List[Dict[str, Any]]