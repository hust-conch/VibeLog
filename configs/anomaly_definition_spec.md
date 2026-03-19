# Anomaly Definition Spec

## Primary Mode: `system_mode`
Judges whether a log indicates a true **system-level** anomaly.

### Abnormal (system-level)
- Explicit system-layer failure with impact evidence:
  - kernel / node / storage / filesystem / network / service-layer failure
  - unrecoverable / corruption / deadlock / panic
  - service unavailable / persistent I/O failure / outage

### Normal (non-system-level)
- App/user/job/process-level failures without system-level impact
- Diagnostic/debug/informational events
- Recovered/corrected/resumed/retry-succeeded transitions
- Ambiguous alert words only (`error/fatal/timeout/warning`) without system evidence

## Auxiliary Mode: `label_aligned_mode`
Benchmark-aligned calibration mode for fair comparison to dataset labels.

### Purpose
- Not the primary method
- Used only for auxiliary reporting and ablation
- Demonstrates semantic-vs-benchmark label gap quantitatively
