from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .policies import PolicyConfig


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def policy_config(config: dict[str, Any]) -> PolicyConfig:
    return PolicyConfig(
        stop_threshold=float(config.get("stop_threshold", 0.55)),
        alpha_conflict=float(config.get("alpha_conflict", 0.35)),
        beta_risk=float(config.get("beta_risk", 0.45)),
        lambda_cost=float(config.get("lambda_cost", 0.10)),
    )
