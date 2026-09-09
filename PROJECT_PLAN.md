# RAMEA Prototype Project Plan

## Objective

Implement and locally validate the Risk-Aware Multimodal Evidence Acquisition (RAMEA) control loop proposed in `multimodal_clinical_research.md`. The prototype starts with a partial patient state, exposes modality-specific evidence tools, estimates uncertainty/conflict/risk/cost, and chooses `RETRIEVE`, `RECONCILE`, `ACT`, or `ESCALATE`.

This first version is a research prototype, not a clinical decision-support system. Synthetic labels and heuristic risk scores are explicitly non-clinical.

## Approved implementation choices

- Language: Python 3.11+.
- AWS client: boto3 using the local AWS profile `aidev`.
- Text model: configurable OpenAI 120B-class Bedrock model; default placeholder is `openai.gpt-oss-120b-1:0` and can be overridden in configuration.
- Multimodal model: configurable lower-cost Bedrock multimodal model; default `amazon.nova-lite-v1:0`.
- Embeddings: configurable Bedrock embedding model; default `amazon.titan-embed-text-v2:0`.
- Orchestration: lightweight custom interfaces, no framework lock-in.
- Default mode: deterministic offline synthetic data; Bedrock is opt-in.
- Initial data progression: synthetic multimodal fixture, then a small OpenI/IU chest X-ray subset, then MIMIC-IV/MIMIC-IV-ED/MIMIC-IV-Note/MIMIC-CXR.

## Milestones

### M0 — Local control-loop smoke test

- Generate deterministic synthetic encounters.
- Implement evidence schemas, timestamps, modality tools, and trajectory logging.
- Run fixed, confidence, and RAMEA policies without an LLM.
- Verify missing-modality and conflict scenarios.

### M1 — Bedrock adapter

- Invoke text and multimodal models through the Bedrock Converse API.
- Add structured-output parsing and retries/timeouts.
- Add optional Titan embedding calls.
- Record model ID, region, token metadata when returned, latency, and errors.
- Never send real restricted MIMIC data to an external API without confirming that the intended data-use agreement permits it.

### M2 — OpenI pilot

- Add an adapter for a manually downloaded, locally stored OpenI image/report subset.
- Validate image/report pairing and provenance.
- Keep synthetic EHR fields separate from real OpenI fields.
- Evaluate image/report agreement and retrieval behavior, not clinical efficacy.

### M3 — MIMIC migration

- Add adapters for MIMIC-IV, MIMIC-IV-ED, MIMIC-IV-Note, and MIMIC-CXR.
- Reconstruct evidence by actual availability time.
- Define the diagnosis taxonomy, endpoint, action-risk rubric, and expert modality-utility protocol before primary experiments.
- Add strict train/validation/test patient-level separation and leakage checks.

### M4 — Research evaluation

- Compare full-context, EHR-only, fixed-retrieval, confidence-based, generic-agent, and RAMEA policies.
- Report diagnosis performance, calibration, retrieval efficiency, premature decisions, conflict detection, unsafe-decision proxies, and risk-weighted information regret.
- Use paired encounters, repeated stochastic runs, bootstrap intervals, effect sizes, and pre-specified multiple-comparison handling.

## Definition of done for the current phase

- Documentation explains data and implementation tasks separately from concepts.
- Offline command `python3 -m scripts.run_prototype` generates and evaluates at least one trajectory.
- Bedrock configuration is explicit but no live call is required for the smoke test.
- Every trajectory includes evidence provenance, timestamps, actions, scores, and final outcome.
- Synthetic risk/action labels are clearly marked as development-only.

## Current offline evaluation command

```bash
python3 -m scripts.evaluate_policies \
  --config configs/prototype.json \
  --output artifacts/policy_comparison.json
```

The current three-case fixture is a wiring smoke test, not a meaningful benchmark. Equal accuracy and retrieval counts are expected until the fixture includes cases where additional modalities change the correct decision or where conflict requires escalation.

## Results archive

All evaluation JSON reports and human-readable interpretations are preserved under `results/`. New evaluator runs default to UTC date-stamped result filenames and should not overwrite prior runs. See `results/README.md` for the naming convention.
