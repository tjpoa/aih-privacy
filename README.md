# AIH – Fall Detection & Privacy-Preserving Baseline

This repository contains the initial baseline pipeline developed in the context of the AIH project.

## Scope
- Signal preprocessing and windowing
- Feature extraction (time-domain)
- Baseline classification model
- Privacy-preserving techniques evaluation

## Structure
- `notebooks/` – exploratory analysis and experiments
- `src/aih_privacy/` – reusable pipeline code
- `docs/` – design decisions and privacy notes
- `scripts/` – runnable pipeline steps

## Architecture

                   ┌───────────────────────────────┐
                   │     SisFall Raw IMU Signal     │
                   │   Acc (x,y,z) + Gyro (x,y,z)   │
                   └───────────────┬───────────────┘
                                   │
                                   ▼
                    ┌───────────────────────────┐
                    │   Preprocessing (common)  │
                    │   - parse filename/labels │
                    │   - optional filtering    │
                    └───────────────┬───────────┘
                                   │
                 ┌─────────────────┴─────────────────┐
                 │                                   │
                 ▼                                   ▼
     ┌─────────────────────────┐         ┌─────────────────────────┐
     │ Accelerometer pipeline  │         │ Gyroscope pipeline       │
     │ - compute magnitude     │         │ - compute magnitude      │
     │   |a| = sqrt(ax²+ay²+az²)│        │   |g| = sqrt(gx²+gy²+gz²)│
     └───────────────┬─────────┘         └───────────────┬─────────┘
                     │                                   │
                     ▼                                   ▼
          ┌──────────────────┐               ┌──────────────────┐
          │ Windowing (ACC)  │               │ Windowing (GYRO) │
          │ fixed size/step  │               │ fixed size/step  │
          └─────────┬────────┘               └─────────┬────────┘
                    │                                  │
                    └───────────────┬──────────────────┘
                                    ▼
                        ┌──────────────────────────┐
                        │ Window alignment / join  │
                        │ same window index, same  │
                        │ label, same subject_id   │
                        └──────────────┬───────────┘
                                       │
         ┌─────────────────────────────┼─────────────────────────────┐
         │                             │                             │
         ▼                             ▼                             ▼
┌──────────────────┐        ┌──────────────────┐           ┌──────────────────┐
│  BASELINE PATH   │        │  DP BEFORE PATH  │           │  DP AFTER PATH   │
│  (no privacy)    │        │ (signal-level)   │           │ (feature-level)  │
└─────────┬────────┘        └─────────┬────────┘           └─────────┬────────┘
          │                           │                              │
          ▼                           ▼                              ▼
┌──────────────────┐      ┌──────────────────┐            ┌──────────────────┐
│ Feature extraction│      │ Laplace DP on    │            │ Feature extraction│
│ ACC feats + GYRO  │      │ ACC+GYRO windows │            │ ACC feats + GYRO  │
│ feats (+ concat)  │      │ (clipping+noise) │            │ feats (+ concat)  │
└─────────┬────────┘      └─────────┬────────┘            └─────────┬────────┘
          │                          ▼                               ▼
          │                ┌──────────────────┐            ┌──────────────────┐
          │                │ Feature extraction│           │ Laplace DP on     │
          │                │ after DP windows  │           │ extracted features│
          │                └─────────┬────────┘            └─────────┬────────┘
          └───────────────┬──────────┴───────────────┬──────────────┘
                          ▼                          ▼
                ┌──────────────────────────────────────┐
                │ Identity Enrichment (subjects table)  │
                │ subject_id, age_group, PID, QIs       │
                └───────────────────┬──────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │ Modeling & Evaluation          │
                    │ fixed subject split            │
                    │ LogReg + RandomForest          │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │ Compare baseline vs DP variants│
                    │ balanced accuracy, F1, reports │
                    └───────────────────────────────┘


## Setup
```bash
cd aih-privacy
.venv/Scripts/activate
python -m pip install -r requirements.txt

