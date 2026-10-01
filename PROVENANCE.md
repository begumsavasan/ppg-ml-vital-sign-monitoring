# Provenance

## Publication

Fingernail-Mounted Photoplethysmography and Machine Learning for Vital-Sign Monitoring in Post-Conflict Karabakh  
DOI: https://doi.org/10.62476/bio.ph.12162

## Development history

During manuscript development, the authoritative package was `revised.zip`, using the repository name `fingernail-ppg-vitals-ml`. Preserved project history identifies the intended analysis entry point as:

```text
src/analyze_ppg_artifacts.py
```

That package expected the PhysioNet CSV dataset under `data/csv/` or via the `PPG_DATA_DIR` environment variable. Raw participant data were intentionally excluded from the repository.

The exact earlier archive is not present in the currently accessible project storage. The implementation now committed here is therefore a **new public reimplementation**, not a byte-identical recovery.

## Published design recovered from the record

The peer-reviewed paper documents:

- subjects: s1, s2, s3
- activities: sit, walk, run
- sampling rate: 500 Hz
- window length: 8 seconds
- total windows: 108
- validation: Leave-One-Subject-Out cross-validation
- model: Random Forest
- PPG-only features: standard deviation, skewness, kurtosis, SNR, spectral entropy, dominant frequency, peak count, amplitude range, HRV-RMSSD, HR estimate
- added accelerometer features: magnitude mean, standard deviation, maximum, spectral entropy

The public script exposes preprocessing and model choices explicitly so that they can be audited rather than silently inferred.

## Dataset

Pulse Transit Time PPG Dataset v1.1.0, PhysioNet  
https://physionet.org/content/pulse-transit-time-ppg/1.1.0/

Raw PhysioNet files are not included in this repository.
