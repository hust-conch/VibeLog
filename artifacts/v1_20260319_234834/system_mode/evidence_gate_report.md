# Evidence Gate Report

- Passed: **True**
- Sample size: 200

| Check | Status | Value | Rule |
|---|---|---|---|
| enough_samples | PASS | `200` | >= 100 |
| output_structure_complete | PASS | `{'evidence_type': 1.0, 'relevance_score': 1.0, 'is_key_evidence': 1.0, 'context_size': 1.0}` | nonnull rate of key fields >= 0.95 |
| protocol_stable | PASS | `{'parse_failed_count': 0, 'fallback_order_mapping_count': 0, 'retry_changed_label_count': 0}` | parse_failed=0, fallback_order_mapping <= max(3,2%), retry_changed_label=0 |
| evidence_not_collapsed | PASS | `{'dominant_type': 'noise_benign_evidence', 'dominant_ratio': 0.34}` | dominant evidence type ratio <= 0.90 |
| key_evidence_has_separation | PASS | `0.195` | 0.10 <= key evidence rate <= 0.40 |

## Evidence Distribution

| Evidence Type | Count |
|---|---:|
| noise_benign_evidence | 68 |
| application_level_evidence | 60 |
| system_level_evidence | 38 |
| context_evidence | 34 |
