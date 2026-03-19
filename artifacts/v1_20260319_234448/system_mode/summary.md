# Experiment Summary

## Overall

| Metric | Value |
|---|---:|
| Accuracy | 0.9350 |
| Precision | 0.4545 |
| Recall | 0.9091 |
| F1 | 0.6061 |
| Samples | 200 |
| Abnormal(True) | 11 |
| Abnormal(Pred) | 22 |
| Unknown Pred | 0 |
| Parse Failed | 0 |
| Fallback Order Mapping | 0 |
| First-Pass Fallback | 4 |
| Keyword Fallback | 0 |
| Retry Used | 4 |
| Retry Changed Label | 0 |
| Key Evidence Count | 62 |
| Key Evidence Rate | 0.3100 |
| System-Level Evidence Count | 20 |

## Confusion Matrix

| TP | FP | FN | TN |
|---:|---:|---:|---:|
| 10 | 12 | 1 | 177 |

## By Dataset

## Evidence Type Distribution

| Evidence Type | Count |
|---|---:|
| noise_benign_evidence | 121 |
| failure_evidence | 36 |
| system_level_evidence | 20 |
| application_level_evidence | 14 |
| recovery_evidence | 8 |
| context_evidence | 1 |

| Dataset | Accuracy | Precision | Recall | F1 | Size |
|---|---:|---:|---:|---:|---:|
| bgl | 0.9350 | 0.4545 | 0.9091 | 0.6061 | 200 |

## Files

- summary_json: `/home/zhangjun/LogPrompt/project/artifacts/v1_20260319_234448/system_mode/summary.json`
- config: `/home/zhangjun/LogPrompt/project/artifacts/v1_20260319_234448/system_mode/config.json`
- predictions: `/home/zhangjun/LogPrompt/project/artifacts/v1_20260319_234448/system_mode/predictions.csv`
- predictions_with_errors: `/home/zhangjun/LogPrompt/project/artifacts/v1_20260319_234448/system_mode/predictions_with_errors.csv`
- false_positives: `/home/zhangjun/LogPrompt/project/artifacts/v1_20260319_234448/system_mode/false_positives.csv`
- false_negatives: `/home/zhangjun/LogPrompt/project/artifacts/v1_20260319_234448/system_mode/false_negatives.csv`
- metrics_by_dataset: `/home/zhangjun/LogPrompt/project/artifacts/v1_20260319_234448/system_mode/metrics_by_dataset.csv`
- error_profile: `/home/zhangjun/LogPrompt/project/artifacts/v1_20260319_234448/system_mode/error_profile.csv`
- evidence_profile: `/home/zhangjun/LogPrompt/project/artifacts/v1_20260319_234448/system_mode/evidence_profile.csv`
- fallback_order_mapping_cases: `/home/zhangjun/LogPrompt/project/artifacts/v1_20260319_234448/system_mode/fallback_order_mapping_cases.csv`
- plot_confusion_matrix: `/home/zhangjun/LogPrompt/project/artifacts/v1_20260319_234448/system_mode/plot_confusion_matrix.png`
- plot_f1_by_dataset: `/home/zhangjun/LogPrompt/project/artifacts/v1_20260319_234448/system_mode/plot_f1_by_dataset.png`
- plot_error_counts: `/home/zhangjun/LogPrompt/project/artifacts/v1_20260319_234448/system_mode/plot_error_counts.png`
