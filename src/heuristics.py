from __future__ import annotations

from .schemas import AgentState


def infer_hypothesis(state: AgentState) -> str:
    text = " ".join(str(item.content).lower() for item in state.visible_evidence)
    if "mild sore throat" in text or "normal oxygen" in text:
        return "viral_uri"
    if "congestion" in text or "bnp" in text or "interstitial" in text:
        return "heart_failure"
    if "d_dimer" in text or "equivocal" in text:
        return "pulmonary_embolism"
    if "consolidation" in text or "productive cough" in text or "crp" in text:
        return "pneumonia"
    return "undifferentiated_dyspnea"


def update_state(state: AgentState) -> None:
    text = " ".join(str(item.content).lower() for item in state.visible_evidence)
    state.hypothesis = infer_hypothesis(state)
    state.uncertainty = max(0.05, 1.0 - 0.22 * len(state.visible_evidence))
    if state.hypothesis == "viral_uri" and "normal oxygen" in text:
        state.uncertainty = 0.12
    modalities = {item.modality for item in state.visible_evidence}
    state.conflict = (
        0.65
        if {"radiology", "image"}.issubset(modalities)
        and state.hypothesis == "pulmonary_embolism"
        else 0.0
    )
    state.risk = {
        "heart_failure": 0.85,
        "pulmonary_embolism": 0.95,
        "pneumonia": 0.55,
        "viral_uri": 0.20,
    }.get(state.hypothesis or "", 0.6)
