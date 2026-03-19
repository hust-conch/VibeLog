# Experiment Summary

## Overall

| Metric | Value |
|---|---:|
| Accuracy | 0.9950 |
| Precision | 0.9697 |
| Recall | 1.0000 |
| F1 | 0.9846 |
| Samples | 200 |
| Abnormal(True) | 32 |
| Abnormal(Pred) | 33 |
| Unknown Pred | 0 |
| Parse Failed | 0 |
| Fallback Order Mapping | 0 |
| First-Pass Fallback | 19 |
| Keyword Fallback | 0 |
| Retry Used | 19 |
| Retry Changed Label | 0 |
| Key Evidence Count | 39 |
| Key Evidence Rate | 0.1950 |
| System-Level Evidence Count | 38 |

## Confusion Matrix

| TP | FP | FN | TN |
|---:|---:|---:|---:|
| 32 | 1 | 0 | 167 |

## By Dataset

## Evidence Type Distribution

| Evidence Type | Count |
|---|---:|
| noise_benign_evidence | 68 |
| application_level_evidence | 60 |
| system_level_evidence | 38 |
| context_evidence | 34 |

| Dataset | Accuracy | Precision | Recall | F1 | Size |
|---|---:|---:|---:|---:|---:|
| thunderbird | 0.9950 | 0.9697 | 1.0000 | 0.9846 | 200 |

## Files

- summary_json: `/home/zhangjun/LogPrompt/project/artifacts/v1_20260319_234834/system_mode/summary.json`
- config: `/home/zhangjun/LogPrompt/project/artifacts/v1_20260319_234834/system_mode/config.json`
- predictions: `/home/zhangjun/LogPrompt/project/artifacts/v1_20260319_234834/system_mode/predictions.csv`
- predictions_with_errors: `/home/zhangjun/LogPrompt/project/artifacts/v1_20260319_234834/system_mode/predictions_with_errors.csv`
- false_positives: `/home/zhangjun/LogPrompt/project/artifacts/v1_20260319_234834/system_mode/false_positives.csv`
- false_negatives: `/home/zhangjun/LogPrompt/project/artifacts/v1_20260319_234834/system_mode/false_negatives.csv`
- metrics_by_dataset: `/home/zhangjun/LogPrompt/project/artifacts/v1_20260319_234834/system_mode/metrics_by_dataset.csv`
- error_profile: `/home/zhangjun/LogPrompt/project/artifacts/v1_20260319_234834/system_mode/error_profile.csv`
- evidence_profile: `/home/zhangjun/LogPrompt/project/artifacts/v1_20260319_234834/system_mode/evidence_profile.csv`
- fallback_order_mapping_cases: `/home/zhangjun/LogPrompt/project/artifacts/v1_20260319_234834/system_mode/fallback_order_mapping_cases.csv`
- plot_confusion_matrix: `/home/zhangjun/LogPrompt/project/artifacts/v1_20260319_234834/system_mode/plot_confusion_matrix.png`
- plot_f1_by_dataset: `/home/zhangjun/LogPrompt/project/artifacts/v1_20260319_234834/system_mode/plot_f1_by_dataset.png`
- plot_error_counts: `/home/zhangjun/LogPrompt/project/artifacts/v1_20260319_234834/system_mode/plot_error_counts.png`
