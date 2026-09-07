# RAMEA: AI and Reinforcement-Learning Concepts Guide

This guide explains the prototype for an AI engineer who understands machine learning and LLMs but is less familiar with reinforcement learning (RL).

## 1. What problem are we solving?

A normal multimodal model receives all available information at once:

```text
EHR + labs + clinical note + radiology report + image -> prediction
```

RAMEA asks a different question:

> Given incomplete information, what evidence should an agent request next, and when is it safe to stop?

That makes the task sequential. The agent does not only predict an answer; it controls access to information over multiple steps.

The prototype is designed to study:

- adaptive modality selection;
- uncertainty-aware retrieval;
- cross-modal disagreement;
- risk-aware stopping;
- escalation when autonomous action is inappropriate;
- retrieval cost and information efficiency.

This is a research prototype. The synthetic labels, rules, and risk values are not clinical guidance.

## 2. Why not just give the model everything?

Full context can improve prediction, but it has several drawbacks:

1. It does not test whether the model knows what evidence is necessary.
2. It may expose information that was not available at the decision time.
3. It increases token, latency, and API cost.
4. It can hide disagreement between modalities inside a single final answer.
5. It provides no natural test of safe stopping or escalation.

The RAMEA experiment treats information acquisition as part of the behavior being evaluated.

## 3. The RL vocabulary mapped to this project

### 3.1 Environment

In RL, the environment is the world an agent interacts with. Here it is a patient encounter plus modality tools.

The environment contains evidence such as:

- structured EHR data;
- laboratory results;
- clinical notes;
- radiology reports;
- medical images.

The agent does not automatically see all of it. The environment reveals evidence after a permitted retrieval action and according to its availability timestamp.

Code location: `src/tools.py`, `src/synthetic_data.py`.

### 3.2 State

The state is the information currently available to the agent.

In this prototype, `AgentState` contains:

- visible evidence;
- current time;
- retrieved modalities;
- current hypothesis;
- uncertainty;
- cross-modal conflict score;
- action-risk estimate.

A simplified state is:

```text
s_t = {
    visible evidence,
    time,
    uncertainty,
    conflict,
    risk,
    retrieved modalities
}
```

The hidden evidence is part of the environment but not part of the agent's visible state.

Code location: `src/schemas.py`.

### 3.3 Action

An action is what the agent does next. RAMEA uses four action types:

- `RETRIEVE`: request another modality;
- `RECONCILE`: inspect or resolve disagreement;
- `ACT`: make the current decision;
- `ESCALATE`: abstain, defer, or require human review.

A retrieval action also selects a modality, for example:

```text
RETRIEVE(labs)
RETRIEVE(radiology)
RETRIEVE(image)
```

### 3.4 Transition

A transition is the environment's response to an action:

```text
(s_t, RETRIEVE(labs)) -> (s_{t+1}, laboratory evidence)
```

The new evidence is added to the visible state. The agent recalculates its hypothesis, uncertainty, conflict, and risk.

The prototype advances the simulated clock to the next available evidence timestamp. This prevents future evidence from appearing before its availability time.

### 3.5 Policy

A policy maps a state to an action:

```text
pi(action | state)
```

In the current prototype, policies are deterministic Python classes:

- `FixedRetrievalPolicy`: follows a fixed modality order;
- `ConfidenceBasedPolicy`: retrieves while confidence is below a threshold;
- `RameaPolicy`: combines uncertainty, conflict, risk, value of information, and cost.

Code location: `src/policies.py`.

### 3.6 Reward or utility

In classical RL, the environment returns a reward after an action. A useful abstract reward for this project could be:

```text
utility = correct decision
        - unsafe decision penalty
        - retrieval cost
        - unnecessary retrieval penalty
        - escalation cost
```

The current prototype does not learn from a reward signal. Instead, it uses a transparent acquisition score:

```text
score(modality) = VOI(modality)
                + alpha * conflict
                + beta * risk
                - lambda * cost(modality)
```

This is intentional. Before training an RL agent, we need to define the state, action space, objective, and evaluation metrics clearly.

### 3.7 Episode

One patient encounter is an episode:

```text
initial state
  -> retrieve labs
  -> retrieve image
  -> reconcile
  -> escalate
  -> terminal outcome
```

The trajectory logger records this complete episode for later analysis.

Code location: `src/schemas.py`, `src/agent.py`.

## 4. Is this currently reinforcement learning?

Not yet.

The current implementation is best described as:

> an offline sequential decision-making prototype with hand-designed policies and heuristic utility estimates.

It has the structure of an RL environment:

- state;
- actions;
- transitions;
- terminal outcomes;
- trajectory logging;
- utility-like scoring.

However, it does not yet have:

- a learned policy;
- a value function;
- a Q-function;
- policy-gradient updates;
- temporal-difference learning;
- an RL training loop;
- validated reward labels.

That separation is important. We should first validate the environment and metrics before training a policy on poorly defined objectives.

## 5. Value of Information in intuitive terms

Suppose the agent is uncertain between two diagnoses. A laboratory result may reduce that uncertainty, while an image may be more useful for a different differential.

The value of information (VOI) asks:

> How much could this evidence improve the eventual decision?

A formal version is:

```text
VOI(m) = expected loss before acquiring m
         - expected loss after acquiring m
```

The prototype uses a simple modality utility table as a placeholder. For example:

```python
{
    "labs": 0.62,
    "radiology": 0.72,
    "image": 0.68,
    "text": 0.45
}
```

These values are not learned or clinically validated. In later work, VOI could be estimated using:

- paired model predictions before and after retrieval;
- calibrated predictive distributions;
- counterfactual evaluation;
- expert annotation;
- learned value functions.

## 6. Why risk changes the decision

Confidence-only behavior treats all decisions similarly. RAMEA distinguishes low-risk and high-risk actions.

For a low-risk case:

```text
moderate confidence + low risk -> ACT may be acceptable
```

For a high-risk case:

```text
moderate confidence + high risk -> RETRIEVE or ESCALATE
```

This is why an agent should not stop solely because its top prediction has a high probability. The consequence of being wrong also matters.

The current risk score is a development heuristic. It is not a probability of patient harm.

## 7. Why conflict matters

Different modalities can disagree:

```text
labs:       supports infection
radiology:  no consolidation
image:      equivocal opacity
history:    supports pneumonia
```

A model that averages everything into one answer may conceal this disagreement. RAMEA makes disagreement an explicit state variable.

When conflict is high, the agent should be more likely to:

- retrieve another relevant modality;
- choose `RECONCILE`;
- escalate rather than act prematurely.

The current conflict detector is deliberately simple. A MIMIC implementation will need a formal contradiction representation and expert-reviewed conflict labels.

## 8. Temporal leakage and partial observability

A dataset may contain a final diagnosis or discharge summary even though that information was written after the decision we are simulating.

If the agent sees it early, evaluation becomes invalid.

The prototype therefore attaches an `available_at` timestamp to evidence and only exposes evidence available at the current simulated time.

This is the main connection to a partially observable decision process:

- the environment contains more information than the agent currently observes;
- actions can reveal additional information;
- the agent must act under incomplete knowledge.

In RL terminology, a fully specified formulation could become a POMDP. We do not need to implement a POMDP solver to benefit from the concept; the key requirement is correct information visibility.

## 9. What the three baselines tell us

### Fixed retrieval

```text
labs -> text -> radiology -> image
```

This measures a non-adaptive tool-use strategy.

### Confidence-only retrieval

```text
if confidence < threshold:
    retrieve
else:
    act
```

This tests whether confidence alone is enough.

### RAMEA

```text
uncertainty + conflict + action risk + VOI - cost
```

This tests whether explicit control signals produce safer or more efficient behavior.

The important comparison is not only raw accuracy. We also examine:

- unnecessary retrievals;
- premature stopping;
- expected modality selection;
- conflict response;
- escalation behavior;
- latency and API cost.

## 10. Why we start with heuristics

Heuristics are useful in this phase because they are:

- deterministic;
- inspectable;
- inexpensive;
- easy to debug;
- independent of LLM prompt variability.

If the controller behaves incorrectly offline, adding an LLM will make diagnosis harder, not solve the fundamental problem.

The intended progression is:

```text
heuristic controller
    -> model-assisted interpretation
    -> learned utility estimates
    -> offline policy learning
    -> carefully evaluated RL or bandit policy
```

## 11. Where Bedrock fits

AWS Bedrock is an optional interpretation and representation layer, not the environment itself.

Configured components are:

- OpenAI 120B-class model through Bedrock for text reasoning;
- Amazon Nova Lite for lower-cost multimodal interpretation;
- Amazon Titan Text Embeddings V2 for semantic representations;
- boto3 with AWS profile `aidev`.

The Bedrock adapter is in `src/bedrock_client.py`.

The default mode is offline because:

- local deterministic runs are cheaper;
- model access may vary by account or region;
- model outputs are stochastic;
- restricted clinical data requires careful data-use review;
- every LLM call should be logged and evaluated separately.

## 12. What a future learned RL version could look like

After the environment and labels are mature, possible approaches include:

### Contextual bandit

Use this when the agent mostly chooses one next modality and receives an immediate utility signal. The state is the current evidence summary, the action is a modality, and the reward measures decision improvement minus cost.

### Offline RL

Use historical or simulated trajectories to learn a policy without interacting with real patients. This requires careful handling of distribution shift and unreliable historical clinician actions.

### Model-based planning

Learn or specify how retrieving a modality changes beliefs, then plan over possible future evidence.

### Policy-gradient or actor-critic methods

Use when the action sequence and delayed utility are central and sufficient training trajectories exist.

For this project, a contextual bandit or offline policy-learning formulation is likely a better first learned approach than immediately using deep RL. The action space is small, and the key challenge is defining trustworthy utility and safety labels.

## 13. Recommended learning order

1. Understand the state/action/transition abstraction.
2. Run the deterministic policies.
3. Inspect trajectory JSONL files.
4. Add scenario-specific evaluation.
5. Add model-assisted interpretation.
6. Define a reward/utility rubric.
7. Compare against expert or counterfactual references.
8. Only then consider bandits or offline RL.

## 14. Useful commands

Run one policy:

```bash
python3 -m scripts.run_prototype --policy ramea
python3 -m scripts.run_prototype --policy confidence
python3 -m scripts.run_prototype --policy fixed
```

Compare all policies:

```bash
python3 -m scripts.evaluate_policies \
  --config configs/prototype.json \
  --output artifacts/policy_comparison.json
```

Validate syntax:

```bash
python3 -m compileall -q src scripts
```

## 15. Current limitations

- Synthetic cases are not clinical data.
- Risk labels are hand-authored.
- VOI values are placeholders.
- The conflict detector is heuristic.
- The diagnosis classifier is keyword-based.
- No LLM call is part of the default execution.
- No learned RL policy exists yet.
- Accuracy on the fixture does not imply clinical performance.

These limitations are features of the prototype stage: they make assumptions visible before we introduce model and data complexity.
