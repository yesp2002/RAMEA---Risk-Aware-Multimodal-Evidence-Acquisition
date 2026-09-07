# Data and Implementation Tasks

This document is the execution checklist for turning the RAMEA proposal into a reproducible prototype and later a MIMIC experiment.

## 1. Data plan

### 1.1 Stage 0: synthetic multimodal fixture

**Purpose:** exercise the entire controller locally without credentialed data or network calls.

Required fields per encounter:

- `encounter_id`, synthetic patient demographics, chief complaint;
- initial vitals and selected initial labs;
- timestamped hidden labs, clinical note, radiology report, and image reference;
- diagnosis label and development-only action-risk label;
- evidence agreement/conflict scenario;
- availability timestamp for every item.

Tasks:

- [ ] Generate deterministic JSONL fixtures with a fixed seed.
- [ ] Include normal, missing-modality, and conflict cases.
- [ ] Keep ground truth separate from agent-visible evidence.
- [ ] Add a data dictionary and provenance field.

### 1.2 Stage 1: OpenI/IU pilot

OpenI provides paired chest X-ray images and radiology reports. Download and use only a small local subset. The paired image/report data will test image retrieval, report retrieval, disagreement, and multimodal interpretation. It is not a complete EHR cohort.

Tasks:

- [ ] Download data through the official/approved source and record version/date.
- [ ] Validate image-report identifiers and duplicate studies.
- [ ] Store raw data outside version control.
- [ ] Create a manifest with local path, source ID, modality, timestamp if available, and license/use notes.
- [ ] Keep any synthetic structured metadata explicitly marked as synthetic.
- [ ] Do not infer clinical labels beyond what the source supports.

### 1.3 Stage 2: MIMIC

Target sources:

- MIMIC-IV: structured hospital/ICU data.
- MIMIC-IV-ED: ED triage and observations.
- MIMIC-IV-Note: clinical notes and radiology reports.
- MIMIC-CXR: chest radiographs and reports.

Tasks:

- [ ] Confirm PhysioNet credentialing, training, and data-use agreements.
- [ ] Define allowed local/API processing before sending any data to Bedrock.
- [ ] Implement patient/encounter/study joins.
- [ ] Reconstruct presentation, initial labs, imaging, report, and endpoint times.
- [ ] Apply availability-time filtering at every agent step.
- [ ] Remove or mask future-result leakage from notes and derived fields.
- [ ] Establish patient-level train/validation/test splits.
- [ ] Document missingness and delayed reports.

## 2. Domain schema tasks

- [ ] Finalize diagnosis taxonomy and endpoint.
- [ ] Finalize `ACT`, `RETRIEVE`, `RECONCILE`, and `ESCALATE` semantics.
- [ ] Define development-only versus clinically validated risk labels.
- [ ] Define conflict severity: none, mild, moderate, severe.
- [ ] Define useful-next-modality reference labels; historical orders are not automatically optimal.
- [ ] Define cost units: tool calls, tokens, latency, and API dollars.
- [ ] Define mandatory structured output fields.

## 3. Implementation tasks

- [x] Pydantic-free standard-library schemas for local smoke testing.
- [x] Timestamp-aware evidence store.
- [x] Modality tool interface.
- [x] Deterministic conflict and risk heuristics for development.
- [x] Fixed and RAMEA controller policies.
- [x] Trajectory logging.
- [x] Offline synthetic fixture generator.
- [x] Configurable Bedrock adapter using boto3 profile `aidev`.
- [ ] Add structured LLM interpretation behind a feature flag.
- [ ] Add Nova Lite image-content requests after local image handling is verified.
- [ ] Add Titan embedding requests and vector-backed evidence search.
- [ ] Add OpenI adapter.
- [ ] Add MIMIC adapters and leakage tests.

## 4. Evaluation tasks

- [ ] Full-context baseline.
- [ ] EHR-only baseline.
- [ ] Fixed retrieval baseline.
- [ ] Confidence threshold baseline.
- [ ] Generic tool-using model baseline.
- [ ] RAMEA policy.
- [ ] Accuracy/macro-F1/AUROC where labels support them.
- [ ] Calibration: ECE and Brier score.
- [ ] Modality retrieval count and information efficiency.
- [ ] Conflict detection precision/recall/F1.
- [ ] Premature decision and unsafe-decision operational definitions.
- [ ] Risk-weighted information regret.
- [ ] Paired bootstrap intervals and repeated stochastic runs.

## 5. Security and reproducibility tasks

- [ ] Never commit credentials, raw restricted data, or generated patient records.
- [ ] Use `AWS_PROFILE=aidev` or explicit boto3 `profile_name="aidev"`; do not embed keys.
- [ ] Default to offline mode.
- [ ] Log model IDs and configuration hashes, not secrets.
- [ ] Redact prompts/responses when data policy requires it.
- [ ] Pin dependencies before collaborative execution.
- [ ] Save the exact fixture seed and configuration with each trajectory.

## 6. Offline comparison milestone

- [x] Confidence-only baseline that ignores risk and conflict.
- [x] JSON configuration loading for policy parameters.
- [x] Comparative summary for fixed, confidence, and RAMEA policies.
- [ ] Add fixture cases where fixed/confidence/RAMEA produce behaviorally different outcomes.
- [ ] Add explicit reference labels for useful next modality and safe stopping.
