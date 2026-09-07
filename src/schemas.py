from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


MODALITIES = ("ehr", "labs", "text", "radiology", "image")
ACTIONS = ("RETRIEVE", "RECONCILE", "ACT", "ESCALATE")


@dataclass(frozen=True)
class Evidence:
    modality: str
    content: dict[str, Any]
    available_at: datetime
    provenance: str
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["available_at"] = self.available_at.isoformat()
        return value


@dataclass
class Encounter:
    encounter_id: str
    initial_state: dict[str, Any]
    hidden_evidence: list[Evidence]
    diagnosis: str
    action_risk: str
    conflict_level: str = "none"
    scenario: str = "standard"
    expected_action: str | None = None
    expected_modality: str | None = None
    requires_additional_evidence: bool = False


@dataclass
class AgentState:
    encounter_id: str
    now: datetime
    visible_evidence: list[Evidence] = field(default_factory=list)
    retrieved_modalities: list[str] = field(default_factory=list)
    uncertainty: float = 1.0
    conflict: float = 0.0
    risk: float = 0.0
    hypothesis: str | None = None
    final_action: str | None = None
    final_diagnosis: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "encounter_id": self.encounter_id,
            "now": self.now.isoformat(),
            "visible_evidence": [item.to_dict() for item in self.visible_evidence],
            "retrieved_modalities": self.retrieved_modalities,
            "uncertainty": self.uncertainty,
            "conflict": self.conflict,
            "risk": self.risk,
            "hypothesis": self.hypothesis,
            "final_action": self.final_action,
            "final_diagnosis": self.final_diagnosis,
        }


@dataclass
class Trajectory:
    encounter_id: str
    policy: str
    events: list[dict[str, Any]] = field(default_factory=list)

    def record(self, event: str, **payload: Any) -> None:
        self.events.append(
            {"event": event, "at": datetime.now().isoformat(), **payload}
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "encounter_id": self.encounter_id,
            "policy": self.policy,
            "events": self.events,
        }
