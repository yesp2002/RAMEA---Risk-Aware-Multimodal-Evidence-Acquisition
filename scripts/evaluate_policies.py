from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.agent import run_encounter
from src.config import load_config, policy_config
from src.evaluation import summarize_all
from src.synthetic_data import build_synthetic_encounters


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare offline RAMEA retrieval policies"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/prototype.json"))
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/policy_comparison.json")
    )
    args = parser.parse_args()
    config = load_config(args.config)
    max_steps = args.max_steps or int(config.get("max_steps", 4))
    encounters = build_synthetic_encounters()
    policy_trajectories_by_policy = {}
    for policy_name in ("fixed", "confidence", "ramea"):
        policy_trajectories_by_policy[policy_name] = [
            run_encounter(encounter, policy_name, max_steps, policy_config(config))
            for encounter in encounters
        ]
    evaluation = summarize_all(encounters, policy_trajectories_by_policy)
    result = {
        "config": config,
        **evaluation,
        "trajectories": {
            policy: [item.to_dict() for item in rows]
            for policy, rows in policy_trajectories_by_policy.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for summary in evaluation["summaries"]:
        diagnostic = summary["diagnostic_performance"]
        acquisition = summary["acquisition"]
        print(
            f"{summary['policy']}: accuracy={diagnostic['accuracy']:.2f}, "
            f"macro_f1={diagnostic['macro_f1']:.2f}, "
            f"mean_retrievals={acquisition['mean_retrievals']:.2f}, "
            f"premature_stop={summary['decision_control']['premature_decision_rate']:.2f}"
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
