from __future__ import annotations

from datetime import datetime, timezone

from .heuristics import update_state
from .policies import (
    ConfidenceBasedPolicy,
    FixedRetrievalPolicy,
    PolicyConfig,
    RameaPolicy,
)
from .schemas import AgentState, Encounter, Evidence, Trajectory
from .tools import ModalityTools


def run_encounter(
    encounter: Encounter,
    policy_name: str = "ramea",
    max_steps: int = 4,
    policy_config: PolicyConfig | None = None,
) -> Trajectory:
    config = policy_config or PolicyConfig()
    if policy_name == "ramea":
        policy = RameaPolicy(config)
    elif policy_name == "confidence":
        policy = ConfidenceBasedPolicy(config.confidence_threshold)
    elif policy_name == "fixed":
        policy = FixedRetrievalPolicy()
    else:
        raise ValueError(f"Unknown policy: {policy_name}")
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
    trajectory = Trajectory(encounter.encounter_id, policy_name)
    trajectory.record(
        "initial_state",
        state=encounter.initial_state,
        diagnosis_ground_truth=encounter.diagnosis,
    )

    for step in range(max_steps):
        update_state(state)
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
        update_state(state)
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
