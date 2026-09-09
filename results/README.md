# Evaluation Results Archive

This directory is the persistent archive for experiment outputs and result interpretations.

## Current archived runs

- `2026-09-07_offline_synthetic_evaluation.json`: deterministic heuristic evaluation on five synthetic encounters.
- `2026-09-07_bedrock_synthetic_evaluation.json`: live Bedrock evaluation using `openai.gpt-oss-120b-1:0` and `amazon.nova-lite-v1:0`.
- `2026-09-07_interpretation.md`: human-readable interpretation of both runs.

## Naming convention

Use one new file per run; do not overwrite an earlier result. The evaluator defaults use a UTC timestamp with seconds:

```text
results/YYYY-MM-DDTHHMMSSZ_<dataset>_<model-or-policy>_<run-purpose>.json
results/YYYY-MM-DDTHHMMSSZ_<dataset>_<model-or-policy>_<run-purpose>.md
```

Examples:

```text
results/2026-09-08_openi_nova-lite_ramea.json
results/2026-09-08_mimic_full_bedrock_comparison.json
results/2026-09-08_mimic_full_bedrock_comparison.md
```

Every JSON result should include:

- configuration and model IDs;
- dataset/scenario version;
- policy names;
- metric summaries;
- paired comparisons where applicable;
- resource telemetry;
- complete trajectories or a linked trajectory artifact;
- explicit proxy/clinical-validation status.

Every Markdown interpretation should include:

- what was evaluated;
- data and model configuration;
- headline metrics;
- comparison with prior runs;
- threshold/gate interpretation;
- limitations and next actions.

## Important status rule

Synthetic outputs are development and controller-validation results. They must not be described as clinical validation. Bedrock calls on synthetic text are LLM evaluations, while the current synthetic PNG attachment only validates the multimodal API path; it is not a meaningful medical-image evaluation.

## Future commands

Prefer explicit result paths for reproducibility:

```bash
python3 -m scripts.evaluate_policies \
  --config configs/prototype.json \
  --output results/YYYY-MM-DD_offline_synthetic_evaluation.json

python3 -m scripts.evaluate_bedrock \
  --config configs/bedrock_synthetic.json \
  --output results/YYYY-MM-DD_bedrock_synthetic_evaluation.json
```
