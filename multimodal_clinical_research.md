# Risk-Aware Multimodal Evidence Acquisition for Clinical Decision-Making Agents

## Abstract

Large language model (LLM)-based agents are increasingly being investigated as systems capable of reasoning over electronic health records (EHRs), interacting with clinical tools, retrieving additional evidence, and supporting multi-step clinical decision-making. However, clinical information is inherently multimodal and incomplete: a patient's state may be represented simultaneously through structured EHR measurements, laboratory results, medications, clinical notes, radiology reports, and medical images. Existing multimodal clinical agents generally evaluate whether models can synthesize these sources when the relevant evidence is already available or when tools are provided within a predefined interaction environment. This leaves an important question insufficiently addressed: **how should an autonomous clinical agent decide which modality to acquire next, whether the available evidence is sufficient, and when conflicting modalities warrant additional investigation or escalation?**

We propose **Risk-Aware Multimodal Evidence Acquisition (RAMEA)**, an agentic framework in which clinical decision-making is formulated as a sequential evidence-acquisition problem under uncertainty. Rather than exposing an agent to the complete patient record, RAMEA initially provides a partial patient state and allows the agent to invoke modality-specific tools for structured EHR data, clinical text, and medical imaging. At each step, the agent estimates clinical uncertainty, cross-modal disagreement, downstream decision risk, and the expected utility and cost of acquiring additional evidence. It then selects among four actions: acquire additional evidence, reconcile conflicting evidence, make a clinical decision, or escalate/abstain.

We propose evaluating RAMEA using linked MIMIC-IV, MIMIC-IV-ED, MIMIC-IV-Note, and MIMIC-CXR data. The evaluation will compare full-context multimodal models, fixed-retrieval agents, confidence-based agents, generic tool-using agents, and the proposed risk-aware agent. In addition to diagnostic performance, we introduce evaluation dimensions for information efficiency, premature decision-making, cross-modal conflict detection, unsafe decisions, and risk-weighted information regret.

The central hypothesis is that **safe multimodal clinical agents should not merely integrate more modalities; they should learn when a particular modality is necessary for a decision and when disagreement or missing evidence makes autonomous action inappropriate**. The proposed framework aims to move multimodal clinical agents from static multimodal prediction toward risk-aware sequential clinical reasoning.

**Keywords:** agentic AI, multimodal clinical AI, electronic health records, medical imaging, clinical decision support, large language models, information acquisition, uncertainty, clinical safety, MIMIC-IV

---

# 1. Introduction

## 1.1 Background

Clinical decision-making is fundamentally a multimodal and sequential process. Clinicians rarely make important decisions from a single source of information. Instead, they integrate structured measurements such as laboratory values and vital signs with longitudinal medical history, medications, clinical narratives, radiology reports, and medical images. Importantly, these sources are not always simultaneously available, complete, or mutually consistent.

Traditional clinical artificial intelligence has largely addressed these information sources through separate models. Predictive models have been trained on structured EHR data, computer vision systems have been developed for medical images, and natural-language-processing systems have extracted information from clinical notes. While these approaches have demonstrated considerable success on individual tasks, they typically operate as specialized prediction systems rather than autonomous systems capable of deciding what information they need next.

The emergence of large language models has changed this paradigm. LLM-based agents can reason over textual information, invoke external tools, retrieve information, interact with simulated environments, and perform multi-step tasks. This has motivated a rapidly growing body of research on agentic healthcare systems.

Recent work demonstrates that LLM agents can navigate EHR environments, perform multi-step clinical tasks, and interact with medical tools. MedAgentBench introduced a virtual FHIR-based EHR environment containing clinically derived tasks and realistic patient records, while FHIR-AgentBench subsequently investigated LLM agents' ability to retrieve and reason over interoperable EHR resources.

More recent work has expanded this paradigm toward autonomous clinical workflows. MIRA, for example, operates within a sandboxed EHR environment and can obtain laboratory, microbiology, and imaging information before generating diagnoses and treatment plans. It was evaluated on more than 500 MIMIC-IV-derived emergency-department cases.

At the same time, multimodal agentic systems have emerged. AgentRx evaluates LLM agents using temporal EHR information, medical images, radiology reports, and clinical notes, while AgentClinic evaluates agents in interactive multimodal clinical environments involving information acquisition and tool use.

These developments suggest that the question is no longer simply whether an LLM can reason over multimodal clinical data.

The more fundamental question is:

> **How should an autonomous clinical agent decide which evidence it needs before making a decision?**

---

## 1.2 The problem with static multimodal reasoning

A conventional multimodal model might receive:

```text
Patient history
+
Laboratory results
+
Vital signs
+
Clinical notes
+
Radiology report
+
Chest X-ray
```

and produce:

```text
Diagnosis: pneumonia
Confidence: 0.87
```

This is useful as a prediction problem, but it does not fully represent clinical decision-making.

Consider a patient presenting with shortness of breath.

The initial information may indicate:

* tachycardia,
* hypoxemia,
* elevated inflammatory markers,
* and a history of cardiopulmonary disease.

The initial evidence may support several competing hypotheses.

The agent may then need to decide whether to:

1. retrieve additional laboratory results;
2. inspect a chest X-ray;
3. retrieve the radiologist's interpretation;
4. examine previous clinical notes;
5. request another diagnostic test;
6. make a recommendation; or
7. escalate because the available evidence is insufficient.

The critical decision is therefore not merely:

[
P(D \mid E)
]

where (D) represents a diagnosis and (E) represents evidence.

Instead, the agent must solve a sequential decision problem:

[
\text{What evidence should I acquire next before deciding what to do?}
]

---

## 1.3 Why confidence alone is insufficient

A naïve agent could use a confidence threshold:

[
\text{Act if } P(D|E) > \tau
]

However, the same confidence level can imply very different safety profiles.

For example:

| Case                | Estimated confidence | Potential action risk | Appropriate behavior  |
| ------------------- | -------------------: | --------------------: | --------------------- |
| Low-risk diagnosis  |                  90% |                   Low | Act                   |
| High-risk diagnosis |                  90% |                  High | Acquire more evidence |
| Low-risk diagnosis  |                  60% |                   Low | Possibly act          |
| High-risk diagnosis |                  60% |                  High | Acquire/escalate      |

Thus, **confidence does not equal safety**.

Recent work on medical abstention demonstrates that high-performing medical LLMs can still fail to abstain appropriately under uncertainty.

Similarly, recent work on Value of Information formalizes the broader problem of deciding whether an agent should act or seek additional information, including medical diagnosis.

These studies motivate our work but do not directly solve the multimodal EHR problem we investigate.

---

## 1.4 Cross-modal disagreement

A second challenge arises because clinical modalities can disagree.

For example:

```text
Structured EHR:
Strong evidence of infection

Laboratory data:
Elevated inflammatory markers

Radiology report:
Possible pulmonary edema

Chest X-ray:
Equivocal

Clinical note:
History consistent with heart failure
```

A simple multimodal model may attempt to fuse all information into a single prediction.

A clinical agent should instead recognize:

> **The evidence sources are in conflict, and the disagreement itself is clinically relevant.**

The agent may therefore need to retrieve additional information before acting.

This motivates a central concept in this work:

### Cross-modal evidence disagreement

We define cross-modal disagreement as the degree to which independently available clinical modalities support inconsistent hypotheses, interpretations, or action recommendations.

Our hypothesis is:

> **Cross-modal disagreement should increase the agent's evidence-acquisition threshold and probability of escalation, especially when the downstream action is high risk.**

---

## 1.5 Research objectives

This study investigates four primary research questions.

### RQ1 — Modality selection

Can an agent learn to select the most clinically useful modality to inspect next rather than retrieving all available information?

### RQ2 — Cross-modal disagreement

Can explicit disagreement detection identify cases in which available modalities provide conflicting evidence?

### RQ3 — Risk-aware decision-making

Does incorporating downstream clinical risk improve the safety of agentic decision-making compared with confidence-only policies?

### RQ4 — Information efficiency

Can a risk-aware agent achieve comparable or improved clinical performance while acquiring substantially less information than a full-context multimodal system?

---

## 1.6 Contributions

We propose five contributions.

1. **A risk-aware multimodal clinical agent framework** that treats evidence acquisition as a sequential decision problem.

2. **A partial-observability multimodal EHR environment** derived from linked MIMIC-IV, MIMIC-IV-ED, MIMIC-IV-Note, and MIMIC-CXR data.

3. **Explicit cross-modal disagreement detection** as a component of clinical agent control.

4. **A risk-aware modality acquisition policy** that jointly considers uncertainty, disagreement, clinical action risk, and information cost.

5. **A safety-oriented evaluation framework** measuring not only diagnostic performance but also premature decisions, unsafe decisions, modality efficiency, and risk-weighted information regret.

---

# 2. Related Work

## 2.1 Clinical AI using EHRs

Electronic health records have traditionally been used for predictive modeling of diagnoses, mortality, readmission, disease progression, and treatment outcomes. Such systems generally define a fixed prediction task and operate over a predefined set of features.

The MIMIC-IV database provides a particularly useful foundation for research in this area. It contains structured hospital and intensive-care information and is linkable to emergency-department, clinical-note, and chest-X-ray resources.

However, conventional EHR prediction assumes that the model receives a predefined feature set. It does not generally give the model control over which information to request.

RAMEA instead treats the EHR as an environment containing multiple information sources that can be queried sequentially.

---

## 2.2 EHR agents and tool use

MedAgentBench introduced a virtual FHIR EHR environment containing 300 clinically derived tasks and realistic patient profiles, allowing medical LLM agents to interact with structured records through standardized interfaces.

FHIR-AgentBench subsequently evaluated realistic EHR question answering using FHIR resources and compared different retrieval and reasoning strategies. It demonstrated that both EHR retrieval and reasoning over complex FHIR structures remain challenging for LLM agents.

These studies establish the importance of tool use in EHR agents.

However, their primary objective is evaluating whether agents can successfully retrieve and reason over information needed for specified tasks.

Our problem is different:

> **Given several potentially useful information sources, which source should the agent acquire next, and when should it stop?**

---

## 2.3 Autonomous clinical agents

MIRA represents a major step toward autonomous clinical agents. It operates in a sandboxed EHR, interacts with a patient agent, and uses numerous tools to request laboratory, microbiology, and imaging information before making diagnostic and therapeutic decisions. It was evaluated using more than 500 MIMIC-IV-derived cases.

MIRA demonstrates that an agent can execute an extended clinical workflow.

Our work does not attempt to reproduce this architecture.

Instead, we focus on the **control policy governing evidence acquisition**:

[
\text{Which information should be acquired before action?}
]

This distinction is important because a system can be capable of invoking many tools without necessarily possessing a principled strategy for deciding which tool call is clinically justified.

---

## 2.4 Dynamic clinical agents

DynamiCare models clinical diagnosis as an interactive multi-round process in which specialist agents query a patient system, integrate newly acquired information, and dynamically modify their composition and strategy. The work uses MIMIC-III-derived MIMIC-Patient data.

This work is closely related to our sequential formulation.

However, DynamiCare emphasizes dynamic multi-agent composition and interaction.

Our work instead emphasizes:

* multimodal evidence,
* modality selection,
* cross-modal disagreement,
* downstream decision risk,
* and information efficiency.

Thus, we do not claim that sequential information gathering itself is novel. Rather, our contribution is the **risk-aware multimodal control problem**.

---

## 2.5 Multimodal clinical agents

AgentRx is particularly relevant to this work. It evaluates LLM agents using multimodal clinical information including temporal EHR data, medical images, radiology reports, and clinical notes. The study compares unimodal and multimodal settings and finds that single-agent approaches can outperform naïve multi-agent configurations while also being better calibrated.

AgentRx establishes an important benchmark for multimodal clinical agents.

However, the primary question is multimodal prediction and comparison of agent configurations.

Our proposed framework instead asks:

> **Should the agent obtain a particular modality at all, and does the value of obtaining that modality depend on clinical risk and disagreement?**

This makes RAMEA complementary to AgentRx rather than a direct replication.

---

## 2.6 Interactive multimodal clinical environments

AgentClinic is a recent open-source benchmark that evaluates multimodal agents in sequential clinical environments with patient interaction, data collection, tools, and medical images. The benchmark includes incomplete-information settings and evaluates agents across multiple specialties and languages.

AgentClinic is particularly important because it demonstrates that static medical question answering can substantially overestimate agent performance in sequential clinical environments.

The authors also identify future work involving more fine-grained hierarchical tools and realistic workflow dependencies and resource constraints.

RAMEA builds on this direction by focusing specifically on **evidence acquisition and modality-level decision control in real-world EHR-linked data**.

---

## 2.7 Agent benchmarking and clinical safety

A 2026 benchmark study of agent systems evaluated AgentClinic and other clinical benchmarks and reported that agentic systems produced only modest performance gains in several settings while increasing computational and latency costs. Multimodal performance remained challenging, highlighting the gap between tool availability and effective tool use.

This motivates our inclusion of information efficiency and computational/tool-use cost as explicit evaluation criteria.

---

## 2.8 Abstention and uncertainty

MedAbstain studies whether medical LLMs appropriately abstain under uncertainty. The study finds that even highly capable models can fail to abstain appropriately and that explicitly allowing abstention can substantially affect model uncertainty.

Our work extends this concept in two ways.

First, abstention is not treated as the only safety mechanism.

The agent has four possible responses:

[
{
\text{retrieve},
\text{reconcile},
\text{act},
\text{escalate}
}
]

Second, the agent can actively reduce uncertainty by acquiring another modality before abstaining.

Thus:

> **RAMEA treats abstention as one possible endpoint of a sequential information-acquisition policy.**

---

## 2.9 Value of Information

Decision-theoretic Value of Information (VoI) provides a principled framework for determining whether obtaining additional information is worthwhile. Recent work applies this idea to LLM agents and medical diagnosis, explicitly considering ambiguity, risk, and the cost of additional communication.

Our work adopts the core intuition of VoI but specializes it to multimodal EHR environments.

In particular, we introduce:

[
\text{Clinical VoI}
===================

## \text{Expected reduction in decision loss}

\text{Evidence acquisition cost}
]

and incorporate cross-modal disagreement and action severity into the acquisition policy.

---

# 3. Research Gap

Based on the literature reviewed above, several capabilities now exist independently:

| Capability                        | Existing research              |
| --------------------------------- | ------------------------------ |
| EHR tool use                      | MedAgentBench, FHIR-AgentBench |
| Autonomous EHR workflows          | MIRA                           |
| Dynamic clinical interaction      | DynamiCare                     |
| Multimodal prediction             | AgentRx                        |
| Multimodal sequential interaction | AgentClinic                    |
| Medical abstention                | MedAbstain                     |
| Value-of-information reasoning    | VoI framework                  |

The gap is therefore **not** the absence of multimodal agents, tool use, dynamic diagnosis, or abstention individually.

The research gap is the intersection:

> **A multimodal clinical agent that explicitly decides which modality to acquire based on expected clinical utility, cross-modal disagreement, downstream action risk, and acquisition cost.**

We formulate this as:

[
\boxed{
\text{Risk-Aware Multimodal Evidence Acquisition}
}
]

---

# 4. Problem Formulation

Let a patient encounter be represented by a set of heterogeneous information sources:

[
\mathcal{M} =
{
M_{EHR},
M_{Lab},
M_{Text},
M_{Image}
}
]

At time (t), the agent observes only a subset:

[
O_t \subseteq \mathcal{M}
]

and maintains a clinical belief state:

[
B_t = P(D,A \mid O_t)
]

where:

* (D) = clinical hypotheses/diagnoses,
* (A) = possible downstream actions,
* (O_t) = currently available evidence.

The agent chooses an action:

[
a_t \in
{
\text{Acquire}(m),
\text{Reconcile},
\text{Act},
\text{Escalate}
}
]

where:

[
m \in \mathcal{M}\setminus O_t
]

If the agent acquires modality (m):

[
O_{t+1}=O_t\cup {m}
]

and the clinical belief state is updated.

The objective is to maximize expected clinical utility:

[
\pi^*
=====

\arg\max_{\pi}
\mathbb{E}
[
U(\pi,O_{1:T})
]
]

where utility incorporates:

* clinical correctness,
* action safety,
* information cost,
* unnecessary retrieval,
* and escalation cost.

---

# 5. Proposed Approach

## 5.1 Overview

We propose **Risk-Aware Multimodal Evidence Acquisition (RAMEA)**.

The architecture contains five components:

1. **Patient State Manager**
2. **Modality Tool Layer**
3. **Multimodal Evidence Interpreter**
4. **Risk and Conflict Estimator**
5. **Evidence Acquisition Controller**

The overall process is:

```text
Initial patient state
        |
        v
Evidence state construction
        |
        v
Clinical hypothesis generation
        |
        v
Risk + uncertainty + conflict estimation
        |
        v
Next evidence decision
   /       |       \
 EHR     Image     Text
   \       |       /
        v
Evidence integration
        |
        v
Risk/conflict reassessment
        |
   +----+----+
   |         |
  Act    Acquire more
   |         |
   |      Escalate
   v         |
Final clinical decision
```

---

# 6. Multimodal EHR Environment

## 6.1 Data sources

The primary experimental environment will combine four linked resources.

### MIMIC-IV

MIMIC-IV provides structured clinical information and can be linked to emergency-department, notes, and chest-X-ray resources.

### MIMIC-IV-ED

MIMIC-IV-ED provides emergency-department observations including triage and vital-sign information and can be linked to MIMIC-IV.

### MIMIC-IV-Note

MIMIC-IV-Note contains deidentified clinical notes, including discharge summaries and a large collection of radiology reports, and is linkable to MIMIC-IV.

### MIMIC-CXR

MIMIC-CXR provides a large collection of chest radiographs and associated radiology reports. The current 2.1 release contains approximately 377,000 images corresponding to more than 227,000 radiographic studies.

---

## 6.2 Patient state

The agent initially receives a restricted representation:

```text
Demographics
+
Chief complaint
+
Initial vital signs
+
Selected initial laboratory results
+
Known medications
+
Selected prior history
```

The following information may initially be hidden:

```text
Additional laboratory results
Radiology report
Chest X-ray
Previous clinical notes
Longitudinal trends
Additional diagnostic information
```

The hidden information becomes accessible through tools.

---

# 7. Agent Tools

The environment exposes modality-specific tools.

## 7.1 Structured EHR tool

```python
get_ehr_history(patient_id)
```

Returns relevant historical information.

---

## 7.2 Laboratory tool

```python
get_laboratory_results(
    patient_id,
    test_group=None,
    time_window=None
)
```

---

## 7.3 Clinical-text tool

```python
get_clinical_notes(
    patient_id,
    note_type=None,
    time_window=None
)
```

---

## 7.4 Radiology-report tool

```python
get_radiology_report(
    patient_id,
    study_type=None
)
```

---

## 7.5 Imaging tool

```python
get_medical_image(
    patient_id,
    modality="CXR",
    study_id=None
)
```

The agent therefore has access to **information acquisition**, not unrestricted access to the entire patient record.

---

# 8. Cross-Modal Evidence Representation

Each retrieved modality is transformed into an evidence representation:

[
E_m = f_m(M_m)
]

where (f_m) is a modality-specific encoder or multimodal model.

The evidence manager maintains:

[
E_t =
{
E_{EHR},
E_{Lab},
E_{Text},
E_{Image}
}
]

along with:

* provenance,
* timestamp,
* confidence,
* modality,
* clinical relevance,
* and contradiction relationships.

This provenance is important because the system should be able to determine not only **what it believes**, but **why it believes it**.

---

# 9. Cross-Modal Conflict Detection

We define a conflict score:

[
C_t =
g(E_1,E_2,\ldots,E_n)
]

The conflict estimator examines whether modalities:

1. support the same hypothesis;
2. support competing hypotheses;
3. differ in severity assessment;
4. contain contradictory factual information; or
5. disagree about the recommended action.

For example:

```text
EHR:
History strongly supports CHF

Lab:
BNP elevated

Image:
No obvious pulmonary edema

Radiology:
Possible pulmonary vascular congestion
```

The agent should not automatically treat one modality as ground truth.

Instead, it should represent:

```text
Evidence agreement: Moderate
Evidence conflict: Moderate
Clinical uncertainty: High
```

and determine whether additional information is necessary.

---

# 10. Risk Estimation

Let (A) denote a potential clinical action.

We define action risk as:

[
R(A)
]

The risk estimator considers:

* potential severity of harm;
* reversibility;
* urgency;
* uncertainty;
* and confidence in supporting evidence.

A low-risk action may tolerate relatively incomplete evidence.

A high-risk action requires stronger evidence.

Therefore:

[
\text{Required Evidence}
\propto
R(A)
]

This principle is central to the framework.

---

# 11. Risk-Aware Modality Selection

For each available modality (m), the agent estimates:

[
VOI(m)
======

\mathbb{E}
[
L_{\text{current}}
------------------

L_{\text{after }m}
]
]

where (L) is expected decision loss.

We then define:

[
Score(m)
========

VOI(m)
+
\alpha C_t
+
\beta R(A)
----------

\lambda Cost(m)
]

The next modality is:

[
m^*
===

\arg\max_m Score(m)
]

The agent retrieves the modality if:

[
Score(m^*) > \tau
]

Otherwise, it either acts or escalates.

---

# 12. Proposed Decision Policy

The complete policy is:

```text
1. Construct current evidence state.

2. Generate candidate diagnoses/actions.

3. Estimate:
   - diagnostic uncertainty,
   - cross-modal disagreement,
   - action risk,
   - evidence sufficiency.

4. Estimate the value of each available modality.

5. Select:
   - retrieve modality,
   - reconcile,
   - act,
   - or escalate.

6. Integrate newly acquired evidence.

7. Repeat until:
   - evidence is sufficient,
   - risk is acceptably low,
   - information value falls below threshold,
   - or escalation is required.
```

---

# 13. Baselines

To establish whether the proposed policy actually contributes value, we will compare against five baselines.

## Baseline 1 — Full-context multimodal model

All available information is provided simultaneously.

```text
EHR + Text + Image
        |
        v
Multimodal LLM
        |
        v
Decision
```

This measures the performance ceiling when information acquisition is removed.

---

## Baseline 2 — EHR-only model

Only structured EHR data is available.

This measures the incremental value of multimodal information.

---

## Baseline 3 — Fixed-retrieval agent

The agent retrieves a predefined sequence:

```text
EHR
→ Labs
→ Text
→ Image
```

This represents a tool-using agent without adaptive modality selection.

---

## Baseline 4 — Confidence-based agent

The agent retrieves another modality only when confidence falls below a fixed threshold:

[
P(D|E)<\tau
]

This directly tests whether confidence alone is sufficient.

---

## Baseline 5 — Generic multimodal agent

A general-purpose agent receives access to all modality tools and is prompted to decide what information to obtain, but without explicit risk or conflict modeling.

This is an important baseline because it tests whether the proposed structured policy adds value beyond prompting a capable LLM to "reason carefully."

---

## Proposed model — RAMEA

RAMEA explicitly incorporates:

[
\text{Uncertainty}
+
\text{Conflict}
+
\text{Action Risk}
+
\text{Information Cost}
]

into its acquisition policy.

---

# 14. Experimental Tasks

## 14.1 Task A — Diagnostic decision

Given a partial patient state, the agent must determine the most likely diagnosis.

Primary metric:

[
Accuracy
]

Secondary metrics:

* AUROC where applicable;
* macro-F1;
* calibration;
* sensitivity for high-risk diagnoses.

---

## 14.2 Task B — Evidence acquisition

The agent must select the next most useful modality.

Evaluation:

[
\text{Selected modality}
\quad vs \quad
\text{expert/reference useful modality}
]

For this task, reference labels can be derived through retrospective clinical review rather than assuming that historically ordered tests were always optimal.

---

## 14.3 Task C — Cross-modal conflict detection

The agent receives deliberately constructed conflicting evidence.

It must determine:

```text
No conflict
Mild conflict
Moderate conflict
Severe conflict
```

and explain which modalities disagree.

---

## 14.4 Task D — Safe stopping

The agent must decide whether available evidence is sufficient.

Possible outputs:

```text
ACT
RETRIEVE
RECONCILE
ESCALATE
```

This directly evaluates the proposed control policy.

---

# 15. Dataset Construction

## 15.1 Encounter selection

We will select encounters for which:

1. sufficient structured EHR data is available;
2. relevant clinical notes are available;
3. a corresponding radiology study is available where appropriate;
4. temporal relationships can be reliably reconstructed;
5. a clinically meaningful endpoint can be defined.

---

## 15.2 Temporal leakage prevention

Temporal leakage is a major methodological concern.

For each patient encounter, information will be divided according to its actual availability time.

For example:

```text
T0:
Presentation

T1:
Initial labs

T2:
Imaging

T3:
Radiology report

T4:
Diagnosis/disposition
```

The agent must not access information from (T_{k+1}) while making a decision at (T_k).

This is essential because MIMIC contains longitudinal information that could otherwise accidentally reveal the final diagnosis.

---

# 16. Missing-Modality Evaluation

We will construct controlled missing-modality scenarios.

### Scenario A

```text
EHR + Text
```

Image unavailable.

### Scenario B

```text
EHR + Image
```

Text unavailable.

### Scenario C

```text
EHR only
```

### Scenario D

```text
EHR + Text + Image
```

complete multimodal evidence.

The objective is not simply to measure accuracy.

We ask:

> **Does the agent recognize when a missing modality is clinically consequential?**

---

# 17. Conflict Perturbation

We will create controlled evidence-conflict cases.

For example:

### Consistent

```text
EHR → pneumonia
Lab → infection
Image → consolidation
Report → pneumonia
```

### Mild conflict

```text
EHR → pneumonia
Lab → infection
Image → equivocal
Report → possible pneumonia
```

### Strong conflict

```text
EHR → pneumonia
Lab → infection
Image → edema
Report → pulmonary edema
```

The agent should increasingly favor:

```text
retrieve → reconcile → escalate
```

as conflict and action risk increase.

---

# 18. Evaluation Metrics

## 18.1 Diagnostic performance

### Accuracy

[
Accuracy =
\frac{\text{correct decisions}}{\text{all decisions}}
]

### Macro-F1

Useful for imbalanced diagnostic categories.

### AUROC

Used for binary clinical outcomes where probability scores are available.

---

# 19. Calibration

A safe agent should not merely be accurate.

Its confidence should correspond to actual correctness.

We will measure:

* Expected Calibration Error (ECE);
* Brier score;
* reliability diagrams.

This is particularly important because AgentRx reports calibration differences between agent configurations.

---

# 20. Information Efficiency

We define:

[
IE =
\frac{\text{clinical utility}}
{\text{number of tool calls}}
]

We will also report:

* number of modalities retrieved;
* number of tool calls;
* tokens consumed;
* latency;
* estimated API cost.

This addresses evidence that current agentic systems can produce only modest accuracy gains while incurring substantially greater resource usage.

---

# 21. Premature Decision Rate

A key proposed metric is:

[
PDR =
\frac{
\text{decisions made before necessary evidence was acquired}
}{
\text{all decisions}
}
]

The challenge is defining "necessary."

We propose using a retrospective expert/reference protocol:

> Would access to an additional modality reasonably be expected to change the clinical decision?

Cases meeting this criterion are labeled as potential premature decisions.

---

# 22. Unsafe Decision Rate

We define:

[
UDR =
\frac{
\text{clinically unsafe decisions}
}{
\text{all decisions}
}
]

Because errors differ in severity, we will also calculate a risk-weighted version:

[
RWUDR =
\frac{
\sum_i w_i I(\text{unsafe}_i)
}{
N
}
]

where (w_i) represents the severity weight of the error.

---

# 23. Cross-Modal Conflict Detection

We evaluate:

* precision;
* recall;
* F1;
* severity classification accuracy.

More importantly, we evaluate whether conflict detection changes behavior appropriately.

For example:

[
P(\text{escalate} \mid \text{high conflict})

>

P(\text{escalate} \mid \text{low conflict})
]

for high-risk cases.

---

# 24. Risk-Weighted Information Regret

We propose a new metric:

## Risk-Weighted Information Regret (RWIR)

For a decision made at time (t), define:

[
RWIR_t =
L(a_t,y)
--------

\min_{a}
L(a,y)
]

weighted by the opportunity to acquire additional information:

[
RWIR =
\sum_t
R(a_t)
\cdot
\Delta L_t
]

where:

* (R(a_t)) = severity/risk of the action;
* (\Delta L_t) = avoidable decision loss that could plausibly have been reduced through additional evidence.

This metric distinguishes between:

> an agent that retrieves one unnecessary test,

and:

> an agent that stops too early before a high-risk decision.

---

# 25. Main Hypotheses

### H1

RAMEA will achieve equal or better clinical performance than generic multimodal agents while requiring fewer modality retrievals.

### H2

Risk-aware acquisition will reduce unsafe decisions compared with confidence-only acquisition.

### H3

Explicit cross-modal conflict detection will reduce premature decisions in conflicting-evidence cases.

### H4

Risk-aware policies will show greater benefits on high-risk cases than on low-risk cases.

### H5

RAMEA will maintain better safety-performance tradeoffs under missing-modality conditions.

### H6

The relationship between information acquisition and clinical accuracy will exhibit diminishing returns, allowing RAMEA to stop earlier in low-risk cases without sacrificing safety.

---

# 26. Results

> **Important:** The following section is intentionally written as a results template. Numerical values must be populated only after running the experiments.

## 26.1 Overall clinical performance

We will compare RAMEA with the full-context multimodal model, EHR-only model, fixed-retrieval agent, confidence-based agent, and generic multimodal agent.

| Model                    |  Accuracy |  Macro-F1 |     AUROC |     ECE ↓ | Tool Calls ↓ | Unsafe Decisions ↓ |
| ------------------------ | --------: | --------: | --------: | --------: | -----------: | -----------------: |
| EHR-only                 |     [TBD] |     [TBD] |     [TBD] |     [TBD] |          N/A |              [TBD] |
| Full-context multimodal  |     [TBD] |     [TBD] |     [TBD] |     [TBD] |            0 |              [TBD] |
| Fixed-retrieval agent    |     [TBD] |     [TBD] |     [TBD] |     [TBD] |        [TBD] |              [TBD] |
| Confidence-based agent   |     [TBD] |     [TBD] |     [TBD] |     [TBD] |        [TBD] |              [TBD] |
| Generic multimodal agent |     [TBD] |     [TBD] |     [TBD] |     [TBD] |        [TBD] |              [TBD] |
| **RAMEA**                | **[TBD]** | **[TBD]** | **[TBD]** | **[TBD]** |    **[TBD]** |          **[TBD]** |

The primary result of interest will not be whether RAMEA obtains the highest raw diagnostic accuracy, but whether it achieves a better **safety–information-efficiency tradeoff**.

---

# 27. Information Acquisition Results

We will analyze how often each agent requests each modality.

| Agent            |       EHR |      Labs | Clinical Text |     Image | Total Retrievals |
| ---------------- | --------: | --------: | ------------: | --------: | ---------------: |
| Fixed retrieval  |     [TBD] |     [TBD] |         [TBD] |     [TBD] |            [TBD] |
| Confidence-based |     [TBD] |     [TBD] |         [TBD] |     [TBD] |            [TBD] |
| Generic agent    |     [TBD] |     [TBD] |         [TBD] |     [TBD] |            [TBD] |
| **RAMEA**        | **[TBD]** | **[TBD]** |     **[TBD]** | **[TBD]** |        **[TBD]** |

We expect RAMEA to acquire different modalities depending on the initial evidence state rather than following a fixed retrieval sequence.

---

# 28. Missing-Modality Results

Performance will be reported under:

1. complete multimodal evidence;
2. missing image;
3. missing text;
4. missing laboratory information;
5. EHR-only conditions.

| Condition                   | Generic Agent | Confidence Agent |     RAMEA |
| --------------------------- | ------------: | ---------------: | --------: |
| Complete                    |         [TBD] |            [TBD] | **[TBD]** |
| Image missing               |         [TBD] |            [TBD] | **[TBD]** |
| Text missing                |         [TBD] |            [TBD] | **[TBD]** |
| Labs missing                |         [TBD] |            [TBD] | **[TBD]** |
| Multiple modalities missing |         [TBD] |            [TBD] | **[TBD]** |

The key outcome is whether the agent recognizes when missing information is **decision-critical**.

---

# 29. Conflict Results

We will compare performance under increasing levels of cross-modal disagreement.

| Conflict Level | Accuracy | Escalation Rate | Unsafe Decisions | Additional Retrievals |
| -------------- | -------: | --------------: | ---------------: | --------------------: |
| None           |    [TBD] |           [TBD] |            [TBD] |                 [TBD] |
| Mild           |    [TBD] |           [TBD] |            [TBD] |                 [TBD] |
| Moderate       |    [TBD] |           [TBD] |            [TBD] |                 [TBD] |
| Severe         |    [TBD] |           [TBD] |            [TBD] |                 [TBD] |

A desirable policy should show:

[
\text{Conflict} \uparrow
\Rightarrow
\text{evidence acquisition/escalation} \uparrow
]

particularly for high-risk decisions.

---

# 30. Risk-Sensitivity Analysis

We will stratify encounters according to action-risk categories.

| Risk     | Confidence-Based | Generic Agent |     RAMEA |
| -------- | ---------------: | ------------: | --------: |
| Low      |            [TBD] |         [TBD] |     [TBD] |
| Moderate |            [TBD] |         [TBD] |     [TBD] |
| High     |            [TBD] |         [TBD] | **[TBD]** |

Our hypothesis is that the largest safety benefit will occur in high-risk cases.

---

# 31. Ablation Study

To determine which components are responsible for improvements, we will remove individual components.

### RAMEA variants

* RAMEA without risk estimation;
* RAMEA without conflict detection;
* RAMEA without information cost;
* RAMEA without uncertainty estimation;
* full RAMEA.

| Variant        |  Accuracy | Unsafe Rate | Tool Calls |      RWIR |
| -------------- | --------: | ----------: | ---------: | --------: |
| No risk        |     [TBD] |       [TBD] |      [TBD] |     [TBD] |
| No conflict    |     [TBD] |       [TBD] |      [TBD] |     [TBD] |
| No cost        |     [TBD] |       [TBD] |      [TBD] |     [TBD] |
| No uncertainty |     [TBD] |       [TBD] |      [TBD] |     [TBD] |
| **Full RAMEA** | **[TBD]** |   **[TBD]** |  **[TBD]** | **[TBD]** |

This experiment is critical because otherwise improvements could simply result from additional tool calls.

---

# 32. Qualitative Case Study

Consider a patient presenting with dyspnea.

The initial EHR indicates:

```text
Tachycardia
Hypoxemia
History of heart failure
Elevated inflammatory markers
```

The agent initially considers:

```text
Pneumonia
Heart failure exacerbation
Pulmonary embolism
```

A naïve agent may immediately request a chest X-ray.

RAMEA estimates:

```text
Diagnostic uncertainty: High
Action risk: High
Cross-modal evidence: Incomplete
Expected image value: High
```

It retrieves the chest X-ray.

The image is equivocal.

The radiology report is then retrieved and suggests pulmonary edema.

The agent updates:

```text
Heart failure: increased probability
Pneumonia: reduced probability
PE: unresolved
```

Because the remaining uncertainty affects a high-risk decision, the agent requests additional evidence rather than immediately acting.

This example illustrates the central mechanism:

> **The agent does not merely use multiple modalities; it decides when each modality is necessary.**

---

# 33. Statistical Analysis

For all primary comparisons, we will report:

* mean and standard deviation across repeated runs where stochastic agents are used;
* bootstrap confidence intervals;
* paired comparisons on identical patient encounters;
* effect sizes;
* and appropriate significance testing.

For accuracy differences, we will use paired bootstrap or permutation-based comparisons where appropriate.

For tool-call counts and cost, non-parametric paired tests may be used because these quantities are likely to be skewed.

Multiple comparisons will be controlled using an appropriate correction such as Benjamini–Hochberg where necessary.

---

# 34. Computational Reproducibility

The implementation will record for each encounter:

```text
Patient encounter ID
Initial observations
Agent action sequence
Tool calls
Retrieved evidence
Evidence timestamps
Agent confidence
Conflict score
Risk estimate
Final decision
Ground truth
Token usage
Latency
```

This creates an auditable trajectory:

```text
Observation
    ↓
Decision
    ↓
Tool call
    ↓
Evidence
    ↓
Updated belief
    ↓
Decision
```

The goal is to make the system reproducible and enable error analysis at the trajectory level rather than only at the final prediction level.

---

# 35. Discussion

## 35.1 Main hypothesis

The central hypothesis of this work is that multimodal clinical intelligence should be evaluated not only by the ability to integrate information but also by the ability to **control information acquisition**.

A model that receives every available modality may achieve strong predictive performance but does not necessarily demonstrate autonomous clinical reasoning.

In contrast, a clinical agent must decide:

> What do I know?

> What do I not know?

> Which missing information matters?

> How costly is it to obtain?

> How dangerous is it to act without it?

> Do different evidence sources agree?

These questions define a more realistic agentic clinical problem.

---

# 36. Why multimodal fusion alone is insufficient

A conventional fusion architecture assumes:

[
E =
E_{EHR}
+
E_{Text}
+
E_{Image}
]

However, this implicitly assumes that all modalities are:

1. available;
2. useful;
3. temporally appropriate;
4. trustworthy;
5. equally relevant to the decision.

These assumptions are unrealistic.

Our framework instead treats each modality as a potential **information acquisition action**.

Thus:

[
\text{Multimodal reasoning}
\neq
\text{Multimodal evidence acquisition}
]

This distinction is the conceptual core of the proposed work.

---

# 37. Clinical safety implications

The proposed approach is not intended to enable autonomous treatment of real patients.

Rather, it investigates a safety mechanism for future clinical decision-support systems.

A clinically useful agent should have the ability to say:

> “I do not have enough evidence.”

But it should also be capable of saying:

> “I know exactly what information I need to reduce that uncertainty.”

This is preferable to either:

1. blindly acting under uncertainty; or
2. indiscriminately abstaining.

The desired behavior is:

[
\boxed{
\text{Uncertainty}
\rightarrow
\text{Targeted evidence acquisition}
\rightarrow
\text{Reassessment}
\rightarrow
\text{Safe decision}
}
]

---

# 38. Relationship to existing agentic systems

RAMEA is not intended to replace existing systems such as MIRA, AgentRx, or AgentClinic.

Instead, it can be viewed as a **control layer** that can be placed around a multimodal clinical agent.

For example:

```text
                    RAMEA
                      |
       +--------------+--------------+
       |              |              |
     MIRA          AgentRx      Other agent
       |              |              |
       +--------------+--------------+
                      |
              Clinical environment
```

This makes the framework potentially compatible with different underlying LLMs and multimodal models.

---

# 39. Limitations

## 39.1 Retrospective data

MIMIC data is retrospective.

The system therefore cannot perfectly reproduce the information available to clinicians at every decision point.

---

## 39.2 Historical clinical decisions are imperfect

Clinician behavior should not be treated as a perfect oracle.

A test being ordered historically does not imply that it was necessarily the optimal next test.

Therefore, the proposed "useful modality" labels should ideally involve expert adjudication.

---

## 39.3 Dataset bias

MIMIC represents clinical practice at a single academic medical center and may not generalize to:

* rural hospitals;
* other countries;
* different demographic populations;
* different healthcare systems.

---

## 39.4 Imaging limitations

The initial study focuses primarily on chest radiography because it is relatively well represented and linkable with MIMIC data.

This limits conclusions about other imaging modalities such as:

* CT;
* MRI;
* ultrasound;
* pathology.

---

## 39.5 Model dependence

Results may depend strongly on the underlying multimodal LLM.

Therefore, experiments should include multiple model families where computationally feasible.

---

## 39.6 Simulated risk labels

Action-risk categories may initially require expert-defined rules rather than direct estimates of actual patient harm.

Future work should investigate clinically validated risk models.

---

# 40. Ethical Considerations

MIMIC data is deidentified and governed by controlled-access procedures and a data-use agreement. MIMIC-CXR and related MIMIC resources are intended for research use and require adherence to the applicable PhysioNet access and data-use requirements.

The proposed study is retrospective and computational.

No clinical deployment or treatment recommendation to real patients is proposed.

Any future prospective evaluation would require:

* institutional review;
* clinical safety oversight;
* prospective validation;
* monitoring for demographic disparities;
* and appropriate human-in-the-loop safeguards.

---

# 41. Expected Contributions

If the hypotheses are supported, the study will demonstrate that:

1. **More modalities are not always better.**
2. **The best next modality depends on the current evidence state.**
3. **Cross-modal disagreement is itself a useful signal.**
4. **Confidence-only stopping policies can be unsafe for high-risk decisions.**
5. **Risk-aware evidence acquisition can improve the safety-efficiency tradeoff.**
6. **Agentic clinical reasoning should be evaluated as a sequential decision process rather than a single prediction.**

---

# 42. Conclusion

This paper proposes Risk-Aware Multimodal Evidence Acquisition (RAMEA), a framework for clinical agents that dynamically acquire and reconcile multimodal EHR evidence before making decisions.

Unlike conventional multimodal clinical models, which assume that relevant information is already available, RAMEA treats clinical evidence acquisition as a sequential decision problem.

The agent must determine:

[
\boxed{
\text{What should I look at next?}
}
]

and:

[
\boxed{
\text{Do I have enough evidence to act safely?}
}
]

The proposed framework combines uncertainty estimation, cross-modal disagreement detection, clinical action risk, and information cost.

We propose evaluating the framework using linked MIMIC-IV, MIMIC-IV-ED, MIMIC-IV-Note, and MIMIC-CXR data under complete, missing-modality, and conflicting-evidence conditions.

The central hypothesis is that the next generation of clinical agents should not simply become better at consuming multimodal information. They should become better at **knowing which information matters, recognizing when evidence conflicts, and knowing when additional evidence is required before action**.

---

# 43. References

1. Al Jorf, B., & Shamout, F. E. (2026). **AgentRx: A Benchmark Study of LLM Agents for Multimodal Clinical Prediction Tasks.** *Proceedings of the 7th Conference on Health, Inference, and Learning*, 333, 52–73.

2. Schmidgall, S., Ziaei, R., Harris, C., Kim, J. W., Reis, E. P., Jopling, J., & Moor, M. (2026). **AgentClinic: A multimodal benchmark for tool-using clinical AI agents.** *npj Digital Medicine, 9*, 499.

3. Liu, Y., Carrero, Z. I., Jiang, X., Ferber, D., Wölflein, G., Zhang, L., et al. (2026). **Benchmarking large language model-based agent systems for clinical decision tasks.** *npj Digital Medicine, 9*, 259.

4. Jiang, Y., Black, K. C., Geng, G., Park, D., Zou, J., & Chen, J. H. (2025). **MedAgentBench: A Realistic Virtual EHR Environment to Benchmark Medical LLM Agents.**

5. Lee, G., Bach, E., Yang, E., Pollard, T., Johnson, A., Choi, E., et al. (2025). **FHIR-AgentBench: Benchmarking LLM Agents for Realistic Interoperable EHR Question Answering.**

6. Shang, T., He, W., Zheng, C., Li, L., Shen, L., & Zhao, B. (2025/2026). **DynamiCare: A Dynamic Multi-Agent Framework for Interactive and Open-Ended Medical Decision-Making.**

7. Ferber, D., et al. (2026). **Towards autonomous medical artificial intelligence agents.** *Nature*.

8. Machcha, S., Yerra, S., Gupta, S., Sahoo, A., Sultana, S., Yu, H., & Yao, Z. (2026). **Knowing When to Abstain: Medical LLMs Under Clinical Uncertainty.** *EACL 2026*.

9. Dong, Y. R., Hu, T., Hui, Z., Zhang, C., Vulić, I., Bobu, A., & Collier, N. (2026). **Value of Information: A Framework for Human-Agent Communication.** *ACL 2026*.

10. Johnson, A. E. W., Bulgarelli, L., Pollard, T., Celi, L. A., Mark, R., & Horng, S. (2024). **MIMIC-IV.** PhysioNet.

11. Johnson, A. E. W., Pollard, T., Mark, R., Berkowitz, S., & Horng, S. (2024). **MIMIC-CXR Database, version 2.1.** PhysioNet.

12. Johnson, A. E. W., Pollard, T., Horng, S., Celi, L. A., & Mark, R. (2023). **MIMIC-IV-Note: Deidentified free-text clinical notes.** PhysioNet.

---

# Appendix A. Proposed Experimental Matrix

| Dimension           | Levels                                       |
| ------------------- | -------------------------------------------- |
| Patient information | Complete / partial                           |
| EHR                 | Available / missing                          |
| Clinical text       | Available / missing                          |
| Imaging             | Available / missing                          |
| Evidence conflict   | None / mild / moderate / severe              |
| Clinical risk       | Low / moderate / high                        |
| Agent policy        | Fixed / confidence / generic / RAMEA         |
| Model               | Multiple LLM/VLM backbones                   |
| Evaluation          | Accuracy / safety / efficiency / calibration |

---

# Appendix B. Proposed Agent Output Schema

```json
{
  "clinical_hypotheses": [
    {
      "condition": "candidate diagnosis",
      "probability": 0.72
    }
  ],
  "uncertainty": 0.31,
  "action_risk": "high",
  "cross_modal_conflict": "moderate",
  "evidence_sufficiency": "insufficient",
  "next_action": "retrieve",
  "next_modality": "chest_xray",
  "reason": "High-risk differential with unresolved cardiopulmonary evidence"
}
```

For the actual experiment, the model's free-form reasoning should not be treated as ground truth. The evaluation should primarily use structured outputs, observable tool calls, and final decisions.

---

# Appendix C. Proposed Main Figure

The main conceptual figure should show:

```text
                  PARTIAL EHR
                      |
       +--------------+--------------+
       |              |              |
      EHR            Text          Labs
       |              |              |
       +--------------+--------------+
                      |
                      v
             Clinical Agent
                      |
          +-----------+-----------+
          |           |           |
      Uncertainty   Conflict     Risk
          |           |           |
          +-----------+-----------+
                      |
                      v
             Evidence Controller
                      |
       +--------------+--------------+
       |              |              |
     Retrieve       Act          Escalate
       |
       v
  Image / Text / EHR
       |
       v
 Cross-modal fusion
       |
       v
 Reassess → Act / Retrieve / Escalate
```

---

# Appendix D. Proposed Main Result Figure

The most important result should ideally be a **safety–efficiency frontier**:

```text
Clinical
Safety
  ^
  |
  |             RAMEA
  |          *
  |       *
  |    *
  |  Generic agent
  | *
  +------------------------> Information / Tool Cost
```

The desired result is not simply the highest accuracy.

The strongest result would demonstrate that RAMEA reaches a favorable point on the **clinical safety versus information acquisition cost frontier**, especially for high-risk and conflicting-evidence cases.
