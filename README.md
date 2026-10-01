# Fingernail-Mounted PPG + Machine Learning

Public companion repository for peer-reviewed bioengineering research on fingernail-mounted photoplethysmography (PPG), physiological signal analysis, and machine-learning-assisted vital-sign monitoring.

## Publication

**Fingernail-Mounted Photoplethysmography and Machine Learning for Vital-Sign Monitoring in Post-Conflict Karabakh**

DOI: https://doi.org/10.62476/bio.ph.12162

## Public analysis workflow

This repository contains a transparent public reimplementation of the paper's **Stage-1 artifact-classification workflow**:

- 8-second PPG windows
- three activity/artifact-proxy classes: sit, walk, run
- 10 PPG-derived features
- 4 accelerometer-derived features
- Random Forest classification
- Leave-One-Subject-Out cross-validation
- comparison of PPG-only versus PPG+accelerometer models

The peer-reviewed feasibility analysis used subjects s1–s3 from the open Pulse Transit Time PPG Dataset and 108 windows in total.

## Provenance

During manuscript development, an authoritative private/local package named `revised.zip` contained a repository `fingernail-ppg-vitals-ml` with the entry point `src/analyze_ppg_artifacts.py`. That exact archive is not present in the currently accessible project storage.

The implementation committed here is therefore a **new public reimplementation reconstructed from the peer-reviewed methods and preserved project history**. It is not claimed to be byte-identical to the original private script.

See [PROVENANCE.md](PROVENANCE.md).

## Dataset

The analysis uses the **Pulse Transit Time PPG Dataset v1.1.0** hosted by PhysioNet:

https://physionet.org/content/pulse-transit-time-ppg/1.1.0/

Raw dataset files are not redistributed here.

## Run

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Place the PhysioNet CSV files in `data/csv/`, or point to the CSV directory with `PPG_DATA_DIR`.

```bash
python src/analyze_ppg_artifacts.py
```

or:

```bash
PPG_DATA_DIR=/path/to/physionet/csv python src/analyze_ppg_artifacts.py
```

Expected filenames are `s1_sit.csv`, `s1_walk.csv`, `s1_run.csv`, and so on.

For an offline smoke test that uses synthetic data only:

```bash
python src/analyze_ppg_artifacts.py --demo-synthetic
```

Outputs are written to `results/`.

## Published reference metrics

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| PPG only | 0.361 ± 0.039 | 0.226 ± 0.083 |
| PPG + accelerometer | 0.787 ± 0.112 | 0.740 ± 0.146 |

These are publication reference values, not hard-coded success criteria for the reconstructed public implementation.

## Scientific boundary

The open-dataset analysis demonstrates feasibility of accelerometer-assisted signal-quality classification. It does **not** directly validate the nail-mounted prototype itself; device-specific bench and field validation remain separate questions.

## Citation

See [CITATION.cff](CITATION.cff).

## Researcher

Begüm Savaşan Asgarlı  
ORCID: https://orcid.org/0000-0003-4809-7115
