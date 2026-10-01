# Data availability and release status

## Publication

DOI: https://doi.org/10.62476/bio.ph.12162

## Dataset

The feasibility analysis uses the open **Pulse Transit Time PPG Dataset v1.1.0** hosted by PhysioNet:

https://doi.org/10.13026/jpan-6n92

The public dataset contains synchronized PPG and inertial channels from 22 healthy participants performing sitting, walking, and running activities. The published feasibility analysis used subjects s1–s3.

Raw PhysioNet CSV files are **not redistributed in this repository**.

## Repository contents

The repository includes:

- a public reimplementation of the Stage-1 artifact-classification workflow;
- citation and provenance metadata;
- an offline synthetic smoke-test mode for code-path verification.

The synthetic mode is not participant data and is not intended to reproduce the paper's numerical results.

## Release criteria

Any additional study-specific material added later must pass:

1. provenance verification;
2. participant privacy and de-identification review;
3. ethics/consent restrictions where applicable;
4. third-party licensing and redistribution review;
5. consistency checks against the published methods and results;
6. removal of local paths, credentials, internal notes, and confidential material.
