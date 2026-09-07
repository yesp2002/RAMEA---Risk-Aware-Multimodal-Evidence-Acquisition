from __future__ import annotations

from dataclasses import dataclass

from .schemas import AgentState


@dataclass(frozen=True)
class PolicyConfig:
    stop_threshold: float = 0.55
    alpha_conflict: float = 0.35
    beta_risk: float = 0.45
    lambda_cost: float = 0.10
    confidence_threshold: float = 0.70


class RameaPolicy:
    modality_voi = {
        "labs": 0.62,
        "text": 0.45,
        "radiology": 0.72,
        "image": 0.68,
        "ehr": 0.30,
    }

    def __init__(self, config: PolicyConfig | None = None) -> None:
        self.config = config or PolicyConfig()

    def choose(
        self, state: AgentState, available: list[str]
    ) -> tuple[str, str | None, float]:
        if not available:
            return ("ESCALATE" if state.risk > 0.75 else "ACT", None, 0.0)
        candidates = {
            modality: self.modality_voi.get(modality, 0.25)
            + self.config.alpha_conflict * state.conflict
            + self.config.beta_risk * state.risk
            - self.config.lambda_cost
            for modality in available
            if modality not in state.retrieved_modalities
        }
        if not candidates:
            return (
                "ESCALATE" if state.risk > 0.75 or state.conflict > 0.5 else "ACT",
                None,
                0.0,
            )
        modality, score = max(candidates.items(), key=lambda pair: pair[1])
        if score > self.config.stop_threshold and (
            state.uncertainty > 0.2 or state.risk > 0.75
        ):
            return "RETRIEVE", modality, round(score, 4)
        if state.conflict > 0.5:
            return "RECONCILE", modality, round(score, 4)
        return "ESCALATE" if state.risk > 0.9 else "ACT", None, round(score, 4)


class ConfidenceBasedPolicy:
    """Retrieves only when uncertainty exceeds a confidence-derived threshold.

    It intentionally ignores action risk and cross-modal conflict so it remains a
    meaningful confidence-only baseline for comparison with RAMEA.
    """

    order = ("labs", "text", "radiology", "image")

    def __init__(self, confidence_threshold: float = 0.70) -> None:
        self.confidence_threshold = confidence_threshold

    def choose(
        self, state: AgentState, available: list[str]
    ) -> tuple[str, str | None, float]:
        confidence = 1.0 - state.uncertainty
        if confidence >= self.confidence_threshold:
            return "ACT", None, round(confidence, 4)
        for modality in self.order:
            if modality in available and modality not in state.retrieved_modalities:
                return "RETRIEVE", modality, round(confidence, 4)
        return "ACT", None, round(confidence, 4)


class FixedRetrievalPolicy:
    order = ("labs", "text", "radiology", "image")

    def choose(
        self, state: AgentState, available: list[str]
    ) -> tuple[str, str | None, float]:
        for modality in self.order:
            if modality in available and modality not in state.retrieved_modalities:
                return "RETRIEVE", modality, 1.0
        return "ACT", None, 0.0
