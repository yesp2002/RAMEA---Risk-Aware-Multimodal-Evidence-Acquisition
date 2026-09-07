from datetime import datetime, timedelta, timezone

from .schemas import Encounter, Evidence


def build_synthetic_encounters() -> list[Encounter]:
    base = datetime(2025, 1, 1, 8, tzinfo=timezone.utc)
    return [
        Encounter(
            encounter_id="synthetic-001",
            initial_state={
                "demographics": {"age": 67, "sex": "F"},
                "chief_complaint": "shortness of breath",
                "vitals": {"temperature_c": 37.9, "heart_rate": 112, "spo2": 91},
            },
            hidden_evidence=[
                Evidence(
                    "labs",
                    {"bnp": 2400, "wbc": 9.2},
                    base + timedelta(minutes=20),
                    "synthetic-labs",
                ),
                Evidence(
                    "radiology",
                    {"impression": "pulmonary vascular congestion"},
                    base + timedelta(minutes=45),
                    "synthetic-report",
                ),
                Evidence(
                    "image",
                    {"finding": "bilateral interstitial opacities"},
                    base + timedelta(minutes=40),
                    "synthetic-cxr",
                ),
            ],
            diagnosis="heart_failure",
            action_risk="high",
            scenario="standard_high_risk",
            expected_action="ESCALATE",
            expected_modality="radiology",
            requires_additional_evidence=True,
        ),
        Encounter(
            encounter_id="synthetic-002",
            initial_state={
                "demographics": {"age": 42, "sex": "M"},
                "chief_complaint": "fever and cough",
                "vitals": {"temperature_c": 39.1, "heart_rate": 104, "spo2": 96},
            },
            hidden_evidence=[
                Evidence(
                    "labs",
                    {"wbc": 15.4, "crp": 12.0},
                    base + timedelta(minutes=18),
                    "synthetic-labs",
                ),
                Evidence(
                    "text",
                    {"note": "productive cough for three days"},
                    base + timedelta(minutes=25),
                    "synthetic-note",
                ),
                Evidence(
                    "radiology",
                    {"impression": "right lower lobe consolidation"},
                    base + timedelta(minutes=50),
                    "synthetic-report",
                ),
            ],
            diagnosis="pneumonia",
            action_risk="moderate",
            scenario="standard_moderate_risk",
            expected_action="ACT",
            expected_modality="labs",
            requires_additional_evidence=True,
        ),
        Encounter(
            encounter_id="synthetic-003",
            initial_state={
                "demographics": {"age": 76, "sex": "F"},
                "chief_complaint": "acute dyspnea",
                "vitals": {"temperature_c": 36.8, "heart_rate": 118, "spo2": 88},
            },
            hidden_evidence=[
                Evidence(
                    "labs",
                    {"d_dimer": 2.8},
                    base + timedelta(minutes=15),
                    "synthetic-labs",
                ),
                Evidence(
                    "radiology",
                    {"impression": "no focal consolidation"},
                    base + timedelta(minutes=35),
                    "synthetic-report",
                ),
                Evidence(
                    "image",
                    {"finding": "equivocal basilar opacity"},
                    base + timedelta(minutes=38),
                    "synthetic-cxr",
                ),
            ],
            diagnosis="pulmonary_embolism",
            action_risk="high",
            conflict_level="moderate",
            scenario="high_risk_conflict",
            expected_action="ESCALATE",
            expected_modality="image",
            requires_additional_evidence=True,
        ),
        Encounter(
            encounter_id="synthetic-004",
            initial_state={
                "demographics": {"age": 76, "sex": "F"},
                "chief_complaint": "acute dyspnea",
                "vitals": {"temperature_c": 36.8, "heart_rate": 118, "spo2": 88},
            },
            hidden_evidence=[
                Evidence(
                    "labs",
                    {"d_dimer": 2.8},
                    base + timedelta(minutes=15),
                    "synthetic-labs",
                ),
                Evidence(
                    "radiology",
                    {"impression": "no focal consolidation"},
                    base + timedelta(minutes=35),
                    "synthetic-report",
                ),
            ],
            diagnosis="pulmonary_embolism",
            action_risk="high",
            scenario="high_risk_missing_image",
            expected_action="ESCALATE",
            expected_modality="image",
            requires_additional_evidence=True,
        ),
        Encounter(
            encounter_id="synthetic-005",
            initial_state={
                "demographics": {"age": 29, "sex": "F"},
                "chief_complaint": "mild sore throat",
                "vitals": {"temperature_c": 36.8, "heart_rate": 76, "spo2": 99},
                "clinical_signal": "normal oxygen",
            },
            hidden_evidence=[
                Evidence(
                    "text",
                    {"note": "self-limited upper respiratory symptoms"},
                    base + timedelta(minutes=30),
                    "synthetic-note",
                ),
            ],
            diagnosis="viral_uri",
            action_risk="low",
            scenario="low_risk_safe_stop",
            expected_action="ACT",
            expected_modality=None,
            requires_additional_evidence=False,
        ),
    ]
