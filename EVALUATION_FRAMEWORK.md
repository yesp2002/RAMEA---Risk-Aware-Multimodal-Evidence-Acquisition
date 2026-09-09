# RAMEA Evaluation Framework

This document maps the evaluation plan in `multimodal_clinical_research.md` to the current implementation.

## Status labels

- **Implemented offline:** computed from the five synthetic encounters and auditable trajectories.
- **Proxy:** uses development-only synthetic labels or heuristics and must not be interpreted as clinical performance.
- **MIMIC required:** needs real encounter labels, expert review, or API telemetry that the current fixture does not contain.

## Implemented metric groups

### Diagnostic performance — implemented offline

- Accuracy.
- Macro-F1.
- High-risk diagnosis sensitivity proxy.
- AUROC proxy for high-risk encounters using the final heuristic risk score.

The AUROC proxy is only meaningful when both positive and negative classes are present. The current five-case fixture is too small for research conclusions.

### Calibration — implemented offline proxy

- Expected Calibration Error (ECE).
- Top-label Brier score proxy.
- Reliability-bin data.

The current confidence is `1 - uncertainty` from a heuristic state estimator. Once an LLM or classifier emits calibrated probabilities, this should be replaced with class-probability calibration.

### Evidence acquisition — implemented offline proxy

- Total retrievals/tool calls.
- Mean retrievals per encounter.
- Unique modalities retrieved.
- Expected-modality retrieval rate using synthetic reference labels.
- Information-efficiency proxy: utility proxy divided by retrieval count.

On MIMIC, expected-next-modality labels should come from expert adjudication or a pre-specified counterfactual protocol. Historical orders alone are not an optimality oracle.

### Missing-modality evaluation — implemented offline proxy

The fixture includes a high-risk case where image evidence is unavailable. The report measures whether the policy escalates instead of silently acting.

MIMIC expansion should evaluate:

- EHR + text;
- EHR + image;
- EHR only;
- complete multimodal evidence;
- naturally missing versus controlled missing modalities.

### Conflict detection — implemented offline proxy

- Conflict precision, recall, and F1.
- Conflict severity accuracy.
- High-conflict escalation rate.
- Low-conflict escalation rate.

The current conflict detector derives a numeric score from visible synthetic evidence. A MIMIC study needs a conflict taxonomy, contradiction rules, natural/perturbed conflict cases, and expert labels.

### Safe stopping and safety — implemented offline proxy

- Expected-action rate.
- Premature Decision Rate (PDR) proxy.
- Unsafe Decision Rate (UDR) proxy.
- Risk-Weighted Unsafe Decision Rate (RWUDR) proxy.
- Final action counts for `ACT`, `RETRIEVE`, `RECONCILE`, and `ESCALATE`.

Current unsafe decisions are operationalized from synthetic reference actions. They are not clinical harm labels.

### Risk-Weighted Information Regret — implemented offline proxy

The report computes a simple risk-weighted regret proxy from:

- diagnosis error;
- premature action;
- synthetic action-risk weight.

The full research metric requires a defensible loss function and an estimate of avoidable loss from additional evidence.

### Resource accounting — partially implemented

The report records:

- tool calls;
- modalities retrieved;
- tokens;
- latency;
- estimated API cost.

The first two are available offline. Tokens, latency, and cost remain `null` until Bedrock calls emit usage and timing metadata.

## Statistical comparison

The evaluator compares policies on identical encounters using:

- paired bootstrap confidence intervals;
- paired sign-permutation p-values;
- paired standardized effect sizes;
- Benjamini–Hochberg adjusted p-values across policy comparisons for each metric.

The current synthetic sample is too small for inferential claims. These methods are included so the same report shape can be used after MIMIC evaluation.

## Reproducibility output

`artifacts/full_evaluation.json` contains:

- configuration;
- per-policy summaries;
- diagnostic/calibration/acquisition/control/conflict/missingness/regret/resource sections;
- scenario breakdowns;
- paired comparisons;
- complete trajectories containing observations, decisions, tool results, timestamps, and final outputs.

The report includes the explicit status:

```text
offline synthetic proxy metrics; not clinical validation
```

## MIMIC implementation checklist

Before treating the metrics as study results:

1. Finalize diagnosis and endpoint taxonomy.
2. Define clinically meaningful action-risk weights.
3. Obtain expert labels for useful next modality.
4. Define conflict severity and disagreement adjudication.
5. Enforce availability-time filtering across all MIMIC tables.
6. Create patient-level train/validation/test splits.
7. Capture Bedrock model usage, latency, and cost metadata.
8. Pre-register primary metrics and comparison hypotheses.
9. Use repeated stochastic runs for LLM policies.
10. Report confidence intervals, effect sizes, corrected comparisons, and subgroup/missingness analyses.

## Threshold interpretation

See `EVALUATION_METRICS_AND_THRESHOLDS.md` for metric definitions, interpretation, prototype starting gates, MIMIC-stage guidance, and warnings against treating synthetic thresholds as clinical acceptance criteria.

## Persistent results

Evaluation outputs and their interpretations are archived under `results/`. The archive includes the offline synthetic run, the live Bedrock synthetic run, and the corresponding interpretation note. See `results/README.md` for versioning and future-run conventions.
