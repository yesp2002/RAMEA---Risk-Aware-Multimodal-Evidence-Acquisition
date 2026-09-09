from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.bedrock_agent import run_bedrock_encounter
from src.bedrock_client import BedrockClient
from src.bedrock_interpreter import BedrockEvidenceInterpreter
from src.config import load_config, policy_config
from src.evaluation import summarize_all
from src.synthetic_data import build_synthetic_encounters


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run live Bedrock evaluation on synthetic RAMEA encounters"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/prototype.json"))
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--policy", choices=("all", "fixed", "confidence", "ramea"), default="all"
    )
    args = parser.parse_args()
    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
        args.output = Path(f"results/{stamp}_bedrock_synthetic_evaluation.json")

    config = load_config(args.config)
    if config.get("mode") == "offline":
        print(
            "WARNING: config mode is offline; this command intentionally performs live Bedrock calls."
        )
    max_steps = args.max_steps or int(config.get("max_steps", 4))
    client = BedrockClient(
        profile_name=config.get("aws_profile", "aidev"),
        region_name=config.get("aws_region", "us-east-1"),
    )
    interpreter = BedrockEvidenceInterpreter(
        client=client,
        text_model_id=config.get("text_model_id", "openai.gpt-oss-120b-1:0"),
        vision_model_id=config.get("vision_model_id", "amazon.nova-lite-v1:0"),
    )
    encounters = build_synthetic_encounters()
    policies = (
        ("fixed", "confidence", "ramea") if args.policy == "all" else (args.policy,)
    )
    trajectories_by_policy = {}
    for policy_name in policies:
        trajectories = []
        for index, encounter in enumerate(encounters, start=1):
            print(
                f"Running Bedrock {policy_name}: encounter {index}/{len(encounters)} ({encounter.encounter_id})"
            )
            trajectories.append(
                run_bedrock_encounter(
                    encounter,
                    interpreter,
                    policy_name,
                    max_steps,
                    policy_config(config),
                )
            )
        trajectories_by_policy[f"bedrock_{policy_name}"] = trajectories

    evaluation = summarize_all(encounters, trajectories_by_policy)
    result = {
        "evaluation_type": "live_bedrock_synthetic",
        "config": config,
        "models": {
            "text": interpreter.text_model_id,
            "vision": interpreter.vision_model_id,
            "aws_profile": config.get("aws_profile", "aidev"),
            "aws_region": config.get("aws_region", "us-east-1"),
        },
        **evaluation,
        "trajectories": {
            policy: [trajectory.to_dict() for trajectory in trajectories]
            for policy, trajectories in trajectories_by_policy.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for summary in evaluation["summaries"]:
        print(
            f"{summary['policy']}: accuracy={summary['diagnostic_performance']['accuracy']:.2f}, "
            f"mean_retrievals={summary['acquisition']['mean_retrievals']:.2f}, "
            f"tokens={summary['resource_usage']['tokens_consumed']}, "
            f"latency_s={summary['resource_usage']['latency_seconds']}"
        )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
