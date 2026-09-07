from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.agent import run_encounter
from src.config import load_config, policy_config
from src.synthetic_data import build_synthetic_encounters


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the offline RAMEA control-loop prototype"
    )
    parser.add_argument(
        "--policy", choices=("ramea", "confidence", "fixed"), default="ramea"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/prototype.json"))
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/trajectories.jsonl")
    )
    args = parser.parse_args()
    config = load_config(args.config)
    max_steps = args.max_steps or int(config.get("max_steps", 4))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for encounter in build_synthetic_encounters():
            trajectory = run_encounter(
                encounter, args.policy, max_steps, policy_config(config)
            )
            handle.write(json.dumps(trajectory.to_dict()) + "\n")
            print(
                f"{encounter.encounter_id}: {args.policy} -> {trajectory.events[-1]['event']}"
            )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
