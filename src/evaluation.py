from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from random import Random
from typing import Any, Iterable

from .schemas import Encounter, Trajectory

RISK_WEIGHTS = {"low": 0.2, "moderate": 0.5, "high": 1.0}
HIGH_RISK_DIAGNOSES = {"heart_failure", "pulmonary_embolism"}


@dataclass(frozen=True)
class MetricSample:
    encounter_id: str
    value: float


def _final(trajectory: Trajectory) -> dict[str, Any]:
    return next(
        (event for event in reversed(trajectory.events) if event["event"] == "final"),
        {},
    )


def _state(trajectory: Trajectory) -> dict[str, Any]:
    return _final(trajectory).get("state", {})


def _retrieved(trajectory: Trajectory) -> list[str]:
    return [
        event.get("modality")
        for event in trajectory.events
        if event.get("event") == "tool_result" and event.get("modality")
    ]


def _confidence(trajectory: Trajectory) -> float:
    return max(0.0, min(1.0, 1.0 - float(_state(trajectory).get("uncertainty", 1.0))))


def _risk_score(trajectory: Trajectory) -> float:
    return float(_state(trajectory).get("risk", 0.0))


def _macro_f1(labels: list[str], predictions: list[str]) -> float:
    classes = sorted(set(labels) | set(predictions))
    if not classes:
        return 0.0
    scores = []
    for label in classes:
        tp = sum(y == label and p == label for y, p in zip(labels, predictions))
        fp = sum(y != label and p == label for y, p in zip(labels, predictions))
        fn = sum(y == label and p != label for y, p in zip(labels, predictions))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    return sum(scores) / len(scores)


def _binary_auc(labels: list[int], scores: list[float]) -> float | None:
    positives = [score for label, score in zip(labels, scores) if label == 1]
    negatives = [score for label, score in zip(labels, scores) if label == 0]
    if not positives or not negatives:
        return None
    wins = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives
        for negative in negatives
    )
    return wins / (len(positives) * len(negatives))


def _calibration(
    confidences: list[float], correct: list[int], bins: int = 10
) -> dict[str, Any]:
    if not confidences:
        return {"ece": 0.0, "brier_score": 0.0, "reliability": []}
    groups: list[list[tuple[float, int]]] = [[] for _ in range(bins)]
    for confidence, outcome in zip(confidences, correct):
        index = min(bins - 1, int(confidence * bins))
        groups[index].append((confidence, outcome))
    reliability = []
    ece = 0.0
    for index, group in enumerate(groups):
        if not group:
            continue
        mean_confidence = sum(item[0] for item in group) / len(group)
        accuracy = sum(item[1] for item in group) / len(group)
        weight = len(group) / len(confidences)
        ece += weight * abs(mean_confidence - accuracy)
        reliability.append(
            {
                "bin": index,
                "count": len(group),
                "confidence": mean_confidence,
                "accuracy": accuracy,
            }
        )
    brier = sum(
        (confidence - outcome) ** 2 for confidence, outcome in zip(confidences, correct)
    ) / len(confidences)
    return {
        "ece": round(ece, 4),
        "brier_score": round(brier, 4),
        "reliability": reliability,
    }


def _conflict_severity(score: float) -> str:
    if score <= 0.0:
        return "none"
    if score < 0.35:
        return "mild"
    if score < 0.75:
        return "moderate"
    return "severe"


def _f1_binary(actual: list[int], predicted: list[int]) -> dict[str, float]:
    tp = sum(a == 1 and p == 1 for a, p in zip(actual, predicted))
    fp = sum(a == 0 and p == 1 for a, p in zip(actual, predicted))
    fn = sum(a == 1 and p == 0 for a, p in zip(actual, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _bootstrap_ci(
    values: list[float], seed: int = 7, iterations: int = 1000
) -> dict[str, float] | None:
    if not values:
        return None
    rng = Random(seed)
    samples = []
    for _ in range(iterations):
        draw = [values[rng.randrange(len(values))] for _ in values]
        samples.append(sum(draw) / len(draw))
    samples.sort()
    low = samples[int(0.025 * (len(samples) - 1))]
    high = samples[int(0.975 * (len(samples) - 1))]
    return {
        "estimate": round(sum(values) / len(values), 4),
        "lower": round(low, 4),
        "upper": round(high, 4),
    }


def _permutation_pvalue(
    values: list[float], seed: int = 11, iterations: int = 2000
) -> float | None:
    if not values:
        return None
    observed = abs(sum(values) / len(values))
    rng = Random(seed)
    extreme = 0
    for _ in range(iterations):
        signed = [value if rng.random() >= 0.5 else -value for value in values]
        if abs(sum(signed) / len(signed)) >= observed:
            extreme += 1
    return round((extreme + 1) / (iterations + 1), 4)


def _effect_size(values: list[float]) -> float | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
    standard_deviation = variance**0.5
    return round(mean / standard_deviation, 4) if standard_deviation else 0.0


def _comparison_statistic(values: list[float]) -> dict[str, Any]:
    return {
        "bootstrap": _bootstrap_ci(values),
        "paired_permutation_p_value": _permutation_pvalue(values),
        "paired_effect_size": _effect_size(values),
    }


def _benjamini_hochberg(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [1.0] * len(p_values)
    running = 1.0
    for rank, (index, p_value) in reversed(list(enumerate(indexed, start=1))):
        running = min(running, p_value * len(p_values) / rank)
        adjusted[index] = round(min(1.0, running), 4)
    return adjusted
    tool_calls = sum(len(_retrieved(trajectory)) for trajectory in trajectories)
    tokens = [
        event.get("usage", {}).get("total_tokens")
        for trajectory in trajectories
        for event in trajectory.events
        if event.get("usage")
    ]
    latencies = [
        event.get("latency_seconds")
        for trajectory in trajectories
        for event in trajectory.events
        if event.get("latency_seconds") is not None
    ]
    costs = [
        event.get("estimated_cost_usd")
        for trajectory in trajectories
        for event in trajectory.events
        if event.get("estimated_cost_usd") is not None
    ]
    return {
        "tool_calls": tool_calls,
        "modalities_retrieved": sum(
            len(set(_retrieved(trajectory))) for trajectory in trajectories
        ),
        "tokens_consumed": sum(tokens) if tokens else None,
        "latency_seconds": round(sum(latencies), 4) if latencies else None,
        "estimated_api_cost_usd": round(sum(costs), 6) if costs else None,
        "llm_resource_accounting_available": bool(tokens or latencies or costs),
    }


def _resource_metrics(trajectories: list[Trajectory]) -> dict[str, Any]:
    tool_calls = sum(len(_retrieved(trajectory)) for trajectory in trajectories)
    tokens = [
        event.get("usage", {}).get("total_tokens")
        for trajectory in trajectories
        for event in trajectory.events
        if event.get("usage")
    ]
    latencies = [
        event.get("latency_seconds")
        for trajectory in trajectories
        for event in trajectory.events
        if event.get("latency_seconds") is not None
    ]
    costs = [
        event.get("estimated_cost_usd")
        for trajectory in trajectories
        for event in trajectory.events
        if event.get("estimated_cost_usd") is not None
    ]
    return {
        "tool_calls": tool_calls,
        "modalities_retrieved": sum(
            len(set(_retrieved(trajectory))) for trajectory in trajectories
        ),
        "tokens_consumed": sum(tokens) if tokens else None,
        "latency_seconds": round(sum(latencies), 4) if latencies else None,
        "estimated_api_cost_usd": round(sum(costs), 6) if costs else None,
        "llm_resource_accounting_available": bool(tokens or latencies or costs),
    }


def summarize(
    encounters: list[Encounter], trajectories: list[Trajectory]
) -> dict[str, Any]:
    by_id = {encounter.encounter_id: encounter for encounter in encounters}
    labels: list[str] = []
    predictions: list[str] = []
    confidences: list[float] = []
    correctness: list[int] = []
    high_risk_labels: list[int] = []
    risk_scores: list[float] = []
    decisions = Counter()
    scenario_counts = Counter()
    action_samples: list[MetricSample] = []
    retrieval_samples: list[MetricSample] = []
    premature_samples: list[MetricSample] = []
    unsafe_samples: list[MetricSample] = []
    regret_samples: list[MetricSample] = []
    expected_actions = expected_modalities = premature_stops = unsafe_decisions = 0
    expected_modality_hits = 0
    conflict_actual: list[int] = []
    conflict_predicted: list[int] = []
    conflict_severity_correct = 0
    conflict_cases = 0
    high_conflict_escalations = 0
    high_conflict_cases = 0
    low_conflict_escalations = 0
    low_conflict_cases = 0
    missing_modality_cases = 0
    missing_modality_recognized = 0

    for trajectory in trajectories:
        encounter = by_id[trajectory.encounter_id]
        scenario_counts[encounter.scenario] += 1
        final = _final(trajectory)
        state = final.get("state", {})
        prediction = final.get("diagnosis", "UNKNOWN")
        action = final.get("action", "UNKNOWN")
        confidence = _confidence(trajectory)
        risk_score = _risk_score(trajectory)
        retrieved = _retrieved(trajectory)
        labels.append(encounter.diagnosis)
        predictions.append(prediction)
        confidences.append(confidence)
        is_correct = int(prediction == encounter.diagnosis)
        correctness.append(is_correct)
        high_risk_labels.append(int(encounter.action_risk == "high"))
        risk_scores.append(risk_score)
        decisions[action] += 1
        action_samples.append(
            MetricSample(
                encounter.encounter_id, float(action == encounter.expected_action)
            )
        )
        retrieval_samples.append(
            MetricSample(encounter.encounter_id, float(len(retrieved)))
        )
        premature = int(encounter.requires_additional_evidence and action == "ACT")
        unsafe = int(
            (
                encounter.action_risk == "high"
                and action == "ACT"
                and encounter.expected_action == "ESCALATE"
            )
            or premature
        )
        premature_samples.append(MetricSample(encounter.encounter_id, float(premature)))
        unsafe_samples.append(MetricSample(encounter.encounter_id, float(unsafe)))
        regret = RISK_WEIGHTS.get(encounter.action_risk, 0.5) * (
            float(not is_correct) + float(premature)
        )
        regret_samples.append(MetricSample(encounter.encounter_id, regret))
        if action == encounter.expected_action:
            expected_actions += 1
        if encounter.expected_modality and encounter.expected_modality in retrieved:
            expected_modalities += 1
            expected_modality_hits += 1
        if premature:
            premature_stops += 1
        if unsafe:
            unsafe_decisions += 1
        actual_conflict = int(encounter.conflict_level != "none")
        observed_score = max(
            (
                float(event.get("state", {}).get("conflict", 0.0))
                for event in trajectory.events
            ),
            default=0.0,
        )
        predicted_conflict = int(observed_score > 0.0)
        conflict_actual.append(actual_conflict)
        conflict_predicted.append(predicted_conflict)
        if actual_conflict:
            conflict_cases += 1
            if _conflict_severity(observed_score) == encounter.conflict_level:
                conflict_severity_correct += 1
            if observed_score >= 0.5:
                high_conflict_cases += 1
                high_conflict_escalations += int(action == "ESCALATE")
        else:
            low_conflict_cases += 1
            low_conflict_escalations += int(action == "ESCALATE")
        if encounter.scenario.startswith("high_risk_missing"):
            missing_modality_cases += 1
            missing_modality_recognized += int(action == "ESCALATE")

    count = len(trajectories)
    calibration = _calibration(confidences, correctness)
    conflict_metrics = _f1_binary(conflict_actual, conflict_predicted)
    high_risk_sensitivity_denominator = sum(high_risk_labels)
    high_risk_correct = sum(
        correct
        for correct, high_risk in zip(correctness, high_risk_labels)
        if high_risk
    )
    utility = (
        sum(1.0 - sample.value for sample in unsafe_samples) / count if count else 0.0
    )
    total_regret = sum(sample.value for sample in regret_samples)
    retrieval_count = sum(sample.value for sample in retrieval_samples)
    return {
        "policy": trajectories[0].policy if trajectories else "unknown",
        "encounters": count,
        "scenario_counts": dict(scenario_counts),
        "diagnostic_performance": {
            "accuracy": round(sum(correctness) / count, 4) if count else 0.0,
            "macro_f1": round(_macro_f1(labels, predictions), 4),
            "high_risk_sensitivity": (
                round(high_risk_correct / high_risk_sensitivity_denominator, 4)
                if high_risk_sensitivity_denominator
                else None
            ),
            "high_risk_auroc_proxy": _binary_auc(high_risk_labels, risk_scores),
        },
        "calibration": calibration,
        "acquisition": {
            "total_retrievals": int(retrieval_count),
            "mean_retrievals": round(retrieval_count / count, 4) if count else 0.0,
            "mean_modalities_retrieved": (
                round(sum(len(set(_retrieved(t))) for t in trajectories) / count, 4)
                if count
                else 0.0
            ),
            "expected_modality_rate": (
                round(expected_modality_hits / count, 4) if count else 0.0
            ),
            "information_efficiency_proxy": (
                round(utility / retrieval_count, 4) if retrieval_count else None
            ),
        },
        "decision_control": {
            "final_actions": dict(decisions),
            "expected_action_rate": (
                round(expected_actions / count, 4) if count else 0.0
            ),
            "premature_decision_rate": (
                round(premature_stops / count, 4) if count else 0.0
            ),
            "unsafe_decision_rate_proxy": (
                round(unsafe_decisions / count, 4) if count else 0.0
            ),
            "risk_weighted_unsafe_rate_proxy": (
                round(
                    sum(
                        sample.value
                        * RISK_WEIGHTS.get(by_id[sample.encounter_id].action_risk, 0.5)
                        for sample in unsafe_samples
                    )
                    / count,
                    4,
                )
                if count
                else 0.0
            ),
            "final_action_counts": dict(decisions),
        },
        "conflict_detection": {
            **conflict_metrics,
            "severity_accuracy": (
                round(conflict_severity_correct / conflict_cases, 4)
                if conflict_cases
                else None
            ),
            "high_conflict_escalation_rate": (
                round(high_conflict_escalations / high_conflict_cases, 4)
                if high_conflict_cases
                else None
            ),
            "low_conflict_escalation_rate": (
                round(low_conflict_escalations / low_conflict_cases, 4)
                if low_conflict_cases
                else None
            ),
        },
        "missing_modality": {
            "cases": missing_modality_cases,
            "recognition_rate_proxy": (
                round(missing_modality_recognized / missing_modality_cases, 4)
                if missing_modality_cases
                else None
            ),
        },
        "risk_weighted_information_regret": {
            "total_proxy": round(total_regret, 4),
            "mean_proxy": round(total_regret / count, 4) if count else 0.0,
        },
        "resource_usage": _resource_metrics(trajectories),
        "scenario_breakdown": _scenario_breakdown(encounters, trajectories),
        "metric_status": "offline synthetic proxy metrics; not clinical validation",
    }


def _scenario_breakdown(
    encounters: list[Encounter], trajectories: list[Trajectory]
) -> dict[str, Any]:
    grouped: dict[str, list[Trajectory]] = defaultdict(list)
    by_id = {encounter.encounter_id: encounter for encounter in encounters}
    for trajectory in trajectories:
        grouped[by_id[trajectory.encounter_id].scenario].append(trajectory)
    result = {}
    for scenario, rows in grouped.items():
        result[scenario] = {
            "encounters": len(rows),
            "final_actions": dict(
                Counter(_final(row).get("action", "UNKNOWN") for row in rows)
            ),
            "retrievals": sum(len(_retrieved(row)) for row in rows),
            "escalations": sum(_final(row).get("action") == "ESCALATE" for row in rows),
        }
    return result


def paired_comparison(
    encounters: list[Encounter],
    trajectories_by_policy: dict[str, list[Trajectory]],
) -> dict[str, Any]:
    policies = list(trajectories_by_policy)
    results = {}
    raw_p_values: dict[str, list[tuple[str, float | None]]] = defaultdict(list)
    for index, left_policy in enumerate(policies):
        for right_policy in policies[index + 1 :]:
            left = {
                row.encounter_id: row for row in trajectories_by_policy[left_policy]
            }
            right = {
                row.encounter_id: row for row in trajectories_by_policy[right_policy]
            }
            accuracy_diff = []
            retrieval_diff = []
            premature_diff = []
            for encounter in encounters:
                left_final = _final(left[encounter.encounter_id])
                right_final = _final(right[encounter.encounter_id])
                accuracy_diff.append(
                    float(right_final.get("diagnosis") == encounter.diagnosis)
                    - float(left_final.get("diagnosis") == encounter.diagnosis)
                )
                retrieval_diff.append(
                    float(
                        len(_retrieved(right[encounter.encounter_id]))
                        - len(_retrieved(left[encounter.encounter_id]))
                    )
                )
                left_premature = (
                    encounter.requires_additional_evidence
                    and left_final.get("action") == "ACT"
                )
                right_premature = (
                    encounter.requires_additional_evidence
                    and right_final.get("action") == "ACT"
                )
                premature_diff.append(float(right_premature) - float(left_premature))
            key = f"{left_policy}_vs_{right_policy}"
            statistics = {
                "accuracy": _comparison_statistic(accuracy_diff),
                "retrievals": _comparison_statistic(retrieval_diff),
                "premature_decision": _comparison_statistic(premature_diff),
            }
            for metric, statistic in statistics.items():
                raw_p_values[metric].append(
                    (key, statistic["paired_permutation_p_value"])
                )
            results[key] = {
                "interpretation": "right policy minus left policy on identical encounters",
                **statistics,
                "paired_encounters": len(encounters),
            }
    for metric, pairs in raw_p_values.items():
        valid = [p_value for _, p_value in pairs if p_value is not None]
        corrected = _benjamini_hochberg(valid)
        corrected_index = 0
        for key, p_value in pairs:
            if p_value is not None:
                results[key][metric]["benjamini_hochberg_adjusted_p_value"] = corrected[
                    corrected_index
                ]
                corrected_index += 1
    return results


def summarize_all(
    encounters: list[Encounter],
    trajectories_by_policy: dict[str, list[Trajectory]],
) -> dict[str, Any]:
    return {
        "summaries": [
            summarize(encounters, trajectories)
            for trajectories in trajectories_by_policy.values()
        ],
        "paired_comparisons": paired_comparison(encounters, trajectories_by_policy),
        "statistical_status": "deterministic synthetic fixture; bootstrap intervals are illustrative only",
    }
