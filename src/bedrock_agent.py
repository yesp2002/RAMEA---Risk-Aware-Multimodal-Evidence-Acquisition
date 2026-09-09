from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .bedrock_interpreter import BedrockEvidenceInterpreter
from .policies import ConfidenceBasedPolicy, FixedRetrievalPolicy, PolicyConfig, RameaPolicy
from .schemas import AgentState, Encounter, Evidence, Trajectory
from .tools import ModalityTools


def _policy(name: str, config: PolicyConfig) -> Any:
    if name == "ramea":
        return RameaPolicy(config)
    if name == "confidence":
        return ConfidenceBasedPolicy(config.confidence_threshold)
    if name == "fixed":
        return FixedRetrievalPolicy()
    raise ValueError(f"Unknown policy: {name}")


def _apply_interpretation(state: AgentState, interpretation: dict[str, Any]) -> None:
    state.hypothesis = interpretation.get("hypothesis", "undifferentiated_dyspnea")
    state.uncertainty = float(interpretation.get("uncertainty", 1.0))
    state.conflict = float(interpretation.get("conflict_score", 0.0))
    risk = interpretation.get("action_risk", "high")
    state.risk = {"low": 0.2, "moderate": 0.55, "high": 0.95}.get(risk, 0.95)


def run_bedrock_encounter(
    encounter: Encounter,
    interpreter: BedrockEvidenceInterpreter,
    policy_name: str = "ramea",
    max_steps: int = 4,
    policy_config: PolicyConfig | None = None,
) -> Trajectory:
    config = policy_config or PolicyConfig()
    policy = _policy(policy_name, config)
    tools = ModalityTools(encounter)
    initial_time = min(
        (item.available_at for item in encounter.hidden_evidence),
        default=datetime.now(timezone.utc),
    )
    state = AgentState(encounter.encounter_id, initial_time, final_diagnosis=None)
    state.visible_evidence.append(
        Evidence(
            modality="ehr",
            content=encounter.initial_state,
            available_at=initial_time,
            provenance="synthetic-initial-state",
        )
    )
    trajectory = Trajectory(encounter.encounter_id, f"bedrock_{policy_name}")
    trajectory.record(
        "initial_state",
        state=encounter.initial_state,
        diagnosis_ground_truth=encounter.diagnosis,
        model_ids={
            "text": interpreter.text_model_id,
            "vision": interpreter.vision_model_id,
        },
    )

    for step in range(max_steps):
        interpretation = interpreter.interpret(state, encounter.initial_state)
        _apply_interpretation(state, interpretation)
        telemetry = interpretation.pop("_telemetry", {})
        trajectory.record(
            "model_interpretation",
            step=step,
            interpretation=interpretation,
            model_id=telemetry.get("model_id"),
            latency_seconds=telemetry.get("latency_seconds"),
            usage=telemetry.get("usage"),
            parse_error=telemetry.get("parse_error"),
            raw_response=telemetry.get("raw_response"),
        )
        available = tools.available_modalities(state.now)
        action, modality, score = policy.choose(state, available)
        trajectory.record(
            "decision",
            step=step,
            action=action,
            modality=modality,
            score=score,
            state=state.to_dict(),
        )
        if action not in {"RETRIEVE", "RECONCILE"} or modality is None:
            state.final_action = action
            state.final_diagnosis = state.hypothesis
            break
        retrieved = tools.retrieve(modality, state.now)
        if not retrieved:
            state.final_action = "ESCALATE"
            trajectory.record("tool_empty", modality=modality)
            break
        state.visible_evidence.extend(retrieved)
        state.retrieved_modalities.append(modality)
        previous_time = state.now
        state.now = max(item.available_at for item in retrieved)
        future_times = [
            item.available_at
            for item in encounter.hidden_evidence
            if item.available_at > previous_time
        ]
        if future_times:
            state.now = min(future_times)
        trajectory.record(
            "tool_result",
            modality=modality,
            evidence=[item.to_dict() for item in retrieved],
        )
    else:
        state.final_action = "ESCALATE"
        state.final_diagnosis = state.hypothesis
        trajectory.record("max_steps", state=state.to_dict())
    trajectory.record(
        "final",
        action=state.final_action,
        diagnosis=state.final_diagnosis,
        state=state.to_dict(),
    )
    return trajectory
