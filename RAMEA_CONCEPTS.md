# RAMEA Concepts

## Partial observability

The agent does not receive the complete patient record. It starts with an initial state and can request additional evidence through modality-specific tools. This makes the task a sequential decision problem rather than a single static prediction.

## Evidence state

The evidence state is the set of currently visible observations plus metadata:

- modality;
- source/provenance;
- availability timestamp;
- relevance;
- confidence;
- contradiction links.

A trajectory must distinguish evidence that exists in the dataset from evidence that was available to the agent at a particular step.

## Actions

The controller chooses one of four development actions:

- `RETRIEVE`: request another modality;
- `RECONCILE`: explicitly inspect disagreement;
- `ACT`: make the current decision;
- `ESCALATE`: abstain or defer because evidence/risk is unacceptable.

## Uncertainty

Uncertainty represents how unresolved the current decision is. The offline prototype uses a transparent heuristic. A later implementation may obtain calibrated probabilities from a model, but a model's verbal confidence is not automatically calibrated uncertainty.

## Cross-modal conflict

Conflict occurs when visible modalities support competing hypotheses, disagree on severity, or contain inconsistent facts. Conflict is not necessarily an error: it is a signal that may justify reconciliation or another retrieval.

## Action risk

Risk is the potential consequence of acting incorrectly, considering severity, urgency, reversibility, and uncertainty. The current risk score is synthetic and intended only to make the controller behavior testable. It is not a validated patient-harm model.

## Value of information

For a candidate modality `m`, value of information (VOI) is the expected reduction in decision loss after acquiring `m`. The prototype approximates this with a transparent modality utility table. A future implementation can estimate VOI using model queries, calibrated predictors, or expert labels.

## Acquisition score

The development policy uses:

```text
score(m) = voi(m) + alpha * conflict + beta * risk - lambda * cost(m)
```

The controller retrieves the best modality only when its score exceeds the stopping threshold. The coefficients must be calibrated on development data and reported with experiments.

## Safe stopping

Stopping is not equivalent to high confidence. A low-risk decision may stop with incomplete evidence, while a high-risk decision may retrieve or escalate despite moderate confidence. This is the central distinction between confidence-only and risk-aware control.

## Temporal leakage

At decision time `t`, the agent may see only records whose availability time is less than or equal to `t`. Future reports, diagnoses, discharge notes, or derived labels must remain hidden even when they are present in the local database.

## Evaluation interpretation

The prototype measures controller mechanics: retrieval behavior, conflict response, stopping, and auditable trajectories. It does not establish clinical safety. MIMIC experiments require a finalized endpoint, clinical taxonomy, expert review protocol, and appropriate data-use controls.

## Confidence-only baseline

The confidence baseline retrieves the next modality only when `1 - uncertainty` is below its threshold. It does not use action risk or cross-modal conflict. Comparing it with RAMEA isolates whether explicit risk/conflict terms change behavior beyond a confidence threshold.
