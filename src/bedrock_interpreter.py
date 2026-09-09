from __future__ import annotations

import json
import struct
import zlib
from typing import Any

from .bedrock_client import BedrockClient
from .schemas import AgentState


DIAGNOSES = ("heart_failure", "pneumonia", "pulmonary_embolism", "viral_uri", "undifferentiated_dyspnea")


def synthetic_png_bytes(seed: str) -> bytes:
    """Create a deterministic non-medical PNG to exercise Nova Lite image transport."""
    digest = zlib.crc32(seed.encode("utf-8"))
    color = (digest & 255, (digest >> 8) & 255, (digest >> 16) & 255)
    width = height = 64
    rows = []
    for y in range(height):
        row = bytearray([0])
        for x in range(width):
            emphasis = 1 if ((x - 32) ** 2 + (y - 32) ** 2) < 20**2 else 0
            row.extend(bytes(min(255, channel + 60 * emphasis) for channel in color))
        rows.append(bytes(row))
    raw = b"".join(rows)

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)

    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


class BedrockEvidenceInterpreter:
    def __init__(self, client: BedrockClient, text_model_id: str, vision_model_id: str) -> None:
        self.client = client
        self.text_model_id = text_model_id
        self.vision_model_id = vision_model_id

    def interpret(self, state: AgentState, initial_state: dict[str, Any]) -> dict[str, Any]:
        evidence = [item.to_dict() for item in state.visible_evidence]
        has_image = any(item.modality == "image" for item in state.visible_evidence)
        model_id = self.vision_model_id if has_image else self.text_model_id
        prompt = self._prompt(state, initial_state, evidence, has_image)
        image_bytes = synthetic_png_bytes(json.dumps(evidence, sort_keys=True)) if has_image else None
        response = self.client.converse(prompt, model_id, image_bytes=image_bytes)
        text = self.client.response_text(response)
        parsed, parse_error = self._parse(text)
        telemetry = response.get("_ramea_telemetry", {})
        usage = telemetry.get("usage", {})
        normalized_usage = {
            "input_tokens": usage.get("inputTokens"),
            "output_tokens": usage.get("outputTokens"),
            "total_tokens": usage.get("totalTokens"),
        }
        parsed["_telemetry"] = {
            "model_id": model_id,
            "modality_mode": "text_and_image" if has_image else "text",
            "latency_seconds": telemetry.get("latency_seconds"),
            "usage": normalized_usage,
            "parse_error": parse_error,
            "raw_response": text,
        }
        return parsed

    @staticmethod
    def _prompt(state: AgentState, initial_state: dict[str, Any], evidence: list[dict[str, Any]], has_image: bool) -> str:
        image_note = "An attached synthetic PNG represents the visible image modality; it is not a real medical image." if has_image else "No image is currently visible."
        return f"""You are an evidence interpretation component in a research prototype. This is synthetic, non-clinical data. Do not provide treatment advice. {image_note}

Current initial state:
{json.dumps(initial_state, indent=2, sort_keys=True)}

Visible evidence:
{json.dumps(evidence, indent=2, sort_keys=True)}

Return ONLY valid JSON with exactly these fields:
{{
  \"hypothesis\": one of {list(DIAGNOSES)},
  \"confidence\": number from 0 to 1,
  \"uncertainty\": number from 0 to 1,
  \"conflict_score\": number from 0 to 1,
  \"conflict_severity\": one of [\"none\", \"mild\", \"moderate\", \"severe\"],
  \"action_risk\": one of [\"low\", \"moderate\", \"high\"],
  \"evidence_sufficiency\": one of [\"sufficient\", \"insufficient\"]
}}
Use only the visible evidence. Do not infer hidden evidence. Confidence and uncertainty should approximately sum to 1."""

    @staticmethod
    def _parse(text: str) -> tuple[dict[str, Any], str | None]:
        candidate = text.strip()
        if candidate.startswith("```"):
            candidate = candidate.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        try:
            value = json.loads(candidate)
            if not isinstance(value, dict):
                raise ValueError("model response was not a JSON object")
        except (json.JSONDecodeError, ValueError, IndexError) as exc:
            return {
                "hypothesis": "undifferentiated_dyspnea",
                "confidence": 0.0,
                "uncertainty": 1.0,
                "conflict_score": 0.0,
                "conflict_severity": "none",
                "action_risk": "high",
                "evidence_sufficiency": "insufficient",
            }, str(exc)
        value["hypothesis"] = value.get("hypothesis", "undifferentiated_dyspnea")
        if value["hypothesis"] not in DIAGNOSES:
            value["hypothesis"] = "undifferentiated_dyspnea"
        for field in ("confidence", "uncertainty", "conflict_score"):
            try:
                value[field] = max(0.0, min(1.0, float(value.get(field, 0.0))))
            except (TypeError, ValueError):
                value[field] = 0.0
        value["conflict_severity"] = value.get("conflict_severity", "none") if value.get("conflict_severity") in {"none", "mild", "moderate", "severe"} else "none"
        value["action_risk"] = value.get("action_risk", "high") if value.get("action_risk") in {"low", "moderate", "high"} else "high"
        value["evidence_sufficiency"] = value.get("evidence_sufficiency", "insufficient") if value.get("evidence_sufficiency") in {"sufficient", "insufficient"} else "insufficient"
        return value, None
