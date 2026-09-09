# Evaluation Interpretation — 2026-09-07

## Runs archived

This interpretation covers the two current result files:

- `2026-09-07_offline_synthetic_evaluation.json`
- `2026-09-07_bedrock_synthetic_evaluation.json`

Both use the same five controlled synthetic encounters and the fixed, confidence-only, and RAMEA policies.

## Offline heuristic run

| Policy | Accuracy | Mean retrievals | Premature-decision proxy | RWUDR proxy | Mean RWIR proxy |
|---|---:|---:|---:|---:|---:|
| Fixed | 1.00 | 2.40 | 0.80 | 0.70 | 0.70 |
| Confidence-only | 1.00 | 2.20 | 0.80 | 0.70 | 0.70 |
| RAMEA | 1.00 | 2.20 | 0.20 | 0.10 | 0.10 |

Interpretation: the deterministic controller behaves as designed. RAMEA escalates selected high-risk cases, reacts to conflict, recognizes the missing high-risk modality, and preserves low-risk stopping. These are controller/mechanics results, not model or clinical results.

## Live Bedrock run

Models:

```text
Text:   openai.gpt-oss-120b-1:0
Vision: amazon.nova-lite-v1:0
Profile: aidev
Region:  us-east-1
```

| Policy | Accuracy | Macro-F1 | Mean retrievals | PDR proxy | RWUDR proxy | Mean RWIR proxy | Tokens | Latency (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Bedrock fixed | 1.00 | 1.00 | 2.40 | 0.80 | 0.70 | 0.70 | 15,899 | 30.7980 |
| Bedrock confidence | 0.80 | 0.67 | 1.00 | 0.80 | 0.70 | 0.80 | 8,404 | 19.2714 |
| Bedrock RAMEA | 1.00 | 1.00 | 2.00 | 0.60 | 0.50 | 0.50 | 13,879 | 26.7611 |

Additional live findings:

- All structured model responses parsed successfully; parse errors: `0`.
- GPT-OSS calls: fixed `14`, confidence `10`, RAMEA `13`.
- Nova Lite calls: fixed `3`, confidence `0`, RAMEA `2`.
- Bedrock RAMEA conflict F1: `0.40`.
- Bedrock RAMEA missing-modality recognition: `0.00`.
- Bedrock RAMEA ECE: `0.204`.
- Bedrock RAMEA high-conflict escalation was not consistently available in the report because the model's conflict score did not reach the configured high-conflict bucket for the controlled case.

## Main conclusion

The live Bedrock result is materially different from the heuristic result. RAMEA retained perfect synthetic diagnosis accuracy and used fewer retrievals than fixed retrieval, but its safety-control behavior degraded substantially:

- PDR increased from `0.20` offline to `0.60` with Bedrock.
- RWUDR increased from `0.10` offline to `0.50` with Bedrock.
- Missing-modality recognition fell from `1.00` offline to `0.00` with Bedrock.
- Conflict F1 fell from `1.00` offline to `0.40` with Bedrock.

This is the first meaningful signal that the hand-designed controller behavior does not automatically transfer when its uncertainty, risk, and conflict inputs come from an LLM.

## Threshold interpretation

Against the prototype gates in `EVALUATION_METRICS_AND_THRESHOLDS.md`:

- Bedrock RAMEA accuracy and Macro-F1 pass numerically, but `n=5` is insufficient for inference.
- Bedrock RAMEA ECE `0.204` fails the `<=0.10` calibration warning gate.
- Bedrock RAMEA PDR `0.60` fails the exploratory `<=0.05` target.
- Bedrock RAMEA RWUDR `0.50` fails the exploratory `<=0.05` target.
- Bedrock RAMEA missing-modality recognition `0.00` fails the `>=0.90` target.
- Bedrock RAMEA conflict F1 `0.40` fails the `>=0.80` exploratory target.

The live run therefore does not pass the current safety-control gates.

## Limitations

1. Only five synthetic encounters were evaluated.
2. Synthetic reference labels are hand-authored.
3. The PNG sent to Nova Lite is a deterministic non-medical fixture. It validates the image-input API path but not medical-image understanding.
4. API dollar cost was not calculated because pricing metadata is not configured.
5. Paired statistical intervals are illustrative only at this sample size.
6. No MIMIC or OpenI clinical data was used.

## Next evaluation changes

- Add real OpenI image/report pairs.
- Separate model interpretation from controller policy tuning.
- Add explicit structured evidence support/contradiction extraction.
- Calibrate model uncertainty before using it in stopping decisions.
- Add repeated Bedrock runs over a larger scenario set.
- Add Bedrock pricing configuration for cost-per-encounter reporting.
