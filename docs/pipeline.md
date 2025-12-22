## Dataset handling

SisFall data are loaded through a dataset-specific loader that:
- parses subject and activity metadata from filenames
- converts raw sensor readings to physical units
- segments signals into fixed, non-overlapping 1s windows (200 samples)

This design allows extending the pipeline to other datasets with minimal changes.

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
