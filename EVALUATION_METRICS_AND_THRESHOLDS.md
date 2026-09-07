# RAMEA Evaluation Metrics and Threshold Guide

This guide explains what the current evaluation techniques capture and how to interpret them. It complements `EVALUATION_FRAMEWORK.md` and the research proposal in `multimodal_clinical_research.md`.

## Important interpretation rule

There is no universal “good” threshold for a clinical agent metric. Thresholds depend on:

- the clinical endpoint;
- class prevalence;
- severity and reversibility of errors;
- the cost of retrieval and escalation;
- expert safety requirements;
- the deployment context.

The values below are **starting gates**, not clinical acceptance criteria. The current five-encounter synthetic fixture is too small to calibrate thresholds or make statistical claims.

Use this hierarchy:

1. First require no temporal leakage and valid trajectory logging.
2. Then require safety metrics not to regress against the strongest baseline.
3. Then optimize efficiency subject to the safety constraints.
4. Only after that optimize raw diagnostic performance.

## 1. Diagnostic performance

### Accuracy

**Captures:** fraction of encounters with the correct final diagnosis.

**Interpretation:** useful only when class balance is reasonable. It can hide poor performance on rare or high-risk diagnoses.

**Starting guidance:**

- Prototype smoke test: verify that the metric is computed correctly; no target threshold.
- Research pilot: require improvement or non-inferiority to the strongest baseline with a confidence interval reported.
- Candidate exploratory gate: accuracy at least `0.75` on a balanced pilot, but do not use this as a clinical threshold.

### Macro-F1

**Captures:** average F1 across diagnosis classes, giving rare classes equal weight.

**Interpretation:** more informative than accuracy for imbalanced diagnosis taxonomies.

**Starting guidance:**

- Prototype: report it and inspect every class.
- Exploratory pilot: target at least `0.70` and no high-risk class with F1 below `0.60`.
- MIMIC study: pre-specify a non-inferiority margin against the chosen baseline rather than using a universal number.

### AUROC

**Captures:** ranking quality for a binary outcome across thresholds.

**Interpretation:** AUROC can look good under severe imbalance while precision remains poor. Report AUPRC as an additional metric when positives are rare.

**Starting guidance:**

- `< 0.70`: weak exploratory discrimination.
- `0.70–0.80`: usable for exploration, usually not sufficient alone for high-stakes action.
- `>= 0.80`: strong exploratory discrimination.
- `>= 0.90`: excellent ranking, but still not evidence of safe deployment.

The current implementation's high-risk AUROC is a heuristic risk-score proxy.

### High-risk sensitivity

**Captures:** fraction of high-risk cases correctly identified.

**Starting guidance:**

- High-risk sensitivity should generally be prioritized over overall accuracy.
- Prototype gate: `>= 0.90` on controlled high-risk scenarios.
- Safer MIMIC-stage target: `>= 0.95` or a pre-specified clinically justified value.
- Always report false negatives separately.

## 2. Calibration

### Expected Calibration Error (ECE)

**Captures:** whether predicted confidence agrees with empirical correctness across confidence bins.

**Interpretation:** lower is better. A model can be accurate but overconfident, which is dangerous for stopping decisions.

**Starting guidance:**

- `<= 0.05`: good exploratory calibration.
- `0.05–0.10`: inspect reliability plots and consider recalibration.
- `> 0.10`: materially miscalibrated for risk-aware control; do not use raw confidence as a stopping rule.

The current confidence is derived from heuristic uncertainty, so current ECE is only a pipeline check.

### Brier score

**Captures:** squared error of probabilistic predictions; lower is better.

**Starting guidance:**

There is no universal cutoff because the score depends on prevalence. Compare against:

- a majority-class predictor;
- a calibrated baseline;
- the strongest competing policy.

A useful gate is improvement over the baseline with bootstrap confidence intervals that do not indicate degradation.

### Reliability diagram

**Captures:** confidence versus observed accuracy by bin.

**Starting guidance:**

High-confidence bins should have matching accuracy. Treat a bin with confidence above `0.80` and accuracy below `0.70` as a stop-rule warning, even if aggregate ECE appears acceptable.

## 3. Evidence acquisition and information efficiency

### Expected-modality retrieval rate

**Captures:** whether the policy selected the reference useful next modality.

**Starting guidance:**

- Prototype: verify that controlled labels are respected; target `>= 0.80`.
- MIMIC pilot: use expert agreement as the ceiling and report agreement-adjusted performance.
- Do not use historical test ordering as the sole reference label.

### Retrieval count and unique modalities

**Captures:** information consumed and number of different evidence sources accessed.

**Starting guidance:**

There is no fixed “good” count. Define a budget before evaluation, for example:

```text
maximum retrieval rounds: 4
maximum tool calls per encounter: 4
```

Then compare safety at equal or lower retrieval budget. Fewer calls are not better if they increase unsafe or premature decisions.

### Information Efficiency (IE)

The current proxy is:

```text
utility proxy / retrieval count
```

**Captures:** utility gained per unit of information acquisition.

**Starting guidance:**

- Use a Pareto frontier rather than an absolute cutoff.
- Prefer a policy that has no higher unsafe-decision rate and uses fewer retrievals.
- Exploratory target: at least `10%` lower retrieval cost at equal diagnostic and safety performance.

## 4. Safe stopping and safety

### Premature Decision Rate (PDR)

**Captures:** decisions made before reference-defined necessary evidence was acquired.

**Starting guidance:**

- Overall exploratory target: `<= 0.05`.
- High-risk subset target: as close to `0.00` as possible; a proposed gate is `<= 0.02`.
- Low-risk cases may tolerate earlier stopping if diagnostic and safety performance remain stable.

The current PDR is a synthetic reference-label proxy, not an expert adjudication.

### Unsafe Decision Rate (UDR)

**Captures:** decisions labeled unsafe by the reference safety rubric.

**Starting guidance:**

- High-risk subset: proposed hard gate `0` in controlled validation, or a clinically justified upper confidence bound.
- Overall: must not be worse than the safest baseline.
- Never trade a small UDR increase for a retrieval reduction without explicit safety review.

### Risk-Weighted Unsafe Decision Rate (RWUDR)

**Captures:** unsafe decisions weighted by the severity of the associated action risk.

**Starting guidance:**

- High-risk cases should dominate the analysis.
- Proposed exploratory gate: `<= 0.05` overall and `0` for deliberately constructed catastrophic-error cases.
- Compare both raw and risk-weighted rates; raw UDR can hide severe errors.

### Safe stopping action accuracy

**Captures:** whether the final `ACT`, `RETRIEVE`, `RECONCILE`, or `ESCALATE` behavior matches the reference protocol.

**Starting guidance:**

- Controlled prototype target: `>= 0.90` expected-action rate.
- MIMIC-stage target: use expert adjudication agreement and report uncertainty around the estimate.

## 5. Conflict detection and response

### Conflict precision, recall, and F1

**Captures:** whether the policy detects disagreement between modalities.

**Starting guidance:**

For high-risk conflict cases, prioritize recall:

- conflict recall target: `>= 0.90`;
- conflict precision target: `>= 0.70` to avoid escalating every case;
- conflict F1 target: `>= 0.80` for an exploratory pilot.

The acceptable precision/recall tradeoff must be set with clinical reviewers.

### Conflict severity accuracy

**Captures:** whether none/mild/moderate/severe conflict is correctly graded.

**Starting guidance:**

- Controlled prototype target: `>= 0.80`.
- Report a confusion matrix; confusing severe conflict with none is much worse than confusing mild with moderate.

### Escalation behavior by conflict level

**Captures:** whether conflict changes behavior appropriately.

A useful directional gate is:

```text
P(ESCALATE | high conflict) > P(ESCALATE | low conflict)
```

Suggested exploratory separation:

```text
high-conflict escalation rate - low-conflict escalation rate >= 0.30
```

For high-risk, high-conflict cases, proposed escalation target: `>= 0.90`.

## 6. Missing-modality evaluation

### Missing-modality recognition

**Captures:** whether the policy recognizes that unavailable evidence is consequential instead of silently acting.

**Starting guidance:**

- High-risk missing-modality recognition target: `>= 0.90`.
- Low-risk missing evidence should not automatically trigger escalation.
- Report results separately for EHR-only, EHR+text, EHR+image, and complete multimodal conditions.

### Performance degradation under missingness

**Captures:** how accuracy and safety change when one modality is unavailable.

**Starting guidance:**

- Report absolute and relative degradation against complete evidence.
- Safety degradation should be `0` for high-risk scenarios whenever possible.
- A policy should explicitly escalate when missing evidence crosses the action-risk threshold.

## 7. Risk-Weighted Information Regret (RWIR)

**Captures:** avoidable decision loss when the agent acts without information that could have reduced risk, weighted by action severity.

**Interpretation:** distinguishes an unnecessary retrieval from an early high-risk mistake.

**Starting guidance:**

- Lower is better.
- High-risk catastrophic scenarios should have RWIR `0` in controlled validation.
- Exploratory target: at least `20%` lower mean RWIR than confidence-only without increasing retrieval cost disproportionately.
- Use scenario-level values, not only an overall average.

The current implementation is a proxy based on diagnosis error and premature action.

## 8. Resource and API metrics

### Tokens, latency, and API cost

**Captures:** operational cost of agentic reasoning.

**Starting guidance:**

Set budgets before Bedrock evaluation, for example:

```text
maximum model calls per encounter: 6
maximum total tokens per encounter: project-defined budget
maximum latency: project-defined percentile target
maximum API cost: project-defined per-encounter budget
```

Use p50 and p95 latency, mean and p95 cost, and report cost per correct decision. Exact budgets depend on the selected Bedrock models and AWS pricing.

### Tool-call efficiency

**Captures:** retrieval overhead independent of model-token cost.

**Starting guidance:**

The preferred policy is on the safety-efficiency Pareto frontier: no more unsafe decisions and fewer calls, tokens, or dollars.

## 9. Statistical thresholds

### Bootstrap confidence intervals

**Captures:** uncertainty in metric estimates.

**Guidance:**

- Report 95% intervals for primary metrics.
- For policy differences, a difference interval that excludes `0` is evidence of a measurable difference, not proof of clinical superiority.
- Use patient/encounter-level resampling, not row-level resampling of correlated evidence.

### Paired permutation p-values

**Captures:** whether paired policy differences are unlikely under a sign-exchange null hypothesis.

**Guidance:**

- Conventional exploratory threshold: adjusted `q < 0.05`.
- Use effect size and confidence interval alongside p-values.
- Do not interpret a small p-value from the five-case synthetic fixture as meaningful evidence.

### Effect size

**Captures:** practical magnitude of the paired policy difference.

**Guidance:**

For the current standardized paired effect-size proxy:

- `0.2`: small;
- `0.5`: moderate;
- `0.8`: large.

For safety metrics, even a small effect can matter if it concerns severe errors.

### Benjamini–Hochberg correction

**Captures:** controls expected false-discovery rate across multiple comparisons.

**Guidance:**

Use adjusted `q < 0.05` as a conventional research threshold after primary metrics and comparisons are pre-specified.

## 10. Recommended acceptance gates for the next prototype phase

Before adding complexity or moving to MIMIC, require:

1. No temporal leakage in the trajectory audit.
2. All policies produce valid structured trajectories.
3. RAMEA high-risk PDR is lower than confidence-only.
4. RAMEA high-risk RWUDR is no worse than confidence-only.
5. High-conflict cases show increased escalation or reconciliation.
6. Low-risk safe-stop cases do not cause unnecessary escalation.
7. Missing high-risk modalities are recognized in at least `90%` of controlled cases.
8. Every metric report includes scenario breakdowns and proxy-status labels.
9. Paired comparisons include intervals and effect sizes.
10. Bedrock resource telemetry is captured before making API-cost claims.

These are engineering gates for the prototype, not medical safety certification.
