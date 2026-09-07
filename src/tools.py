from __future__ import annotations

from datetime import datetime

from .schemas import Encounter, Evidence


class ModalityTools:
    """Expose only evidence available at the requested time."""

    def __init__(self, encounter: Encounter) -> None:
        self.encounter = encounter

    def available_modalities(self, now: datetime) -> list[str]:
        return [
            modality
            for modality in {item.modality for item in self.encounter.hidden_evidence}
            if any(item.modality == modality and item.available_at <= now for item in self.encounter.hidden_evidence)
        ]

    def retrieve(self, modality: str, now: datetime) -> list[Evidence]:
        if modality not in self.available_modalities(now):
            return []
        return [
            item
            for item in self.encounter.hidden_evidence
            if item.modality == modality and item.available_at <= now
        ]
