┌───────────────────────────────┐
│ SisFall raw files (.txt)      │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Parse filename metadata       │
│ subject_id, age_group, label  │
│ activity_code                 │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Identity Enrichment (EARLY)   │
│ Merge with subjects_df        │
│ -> file_registry_df           │
└───────────────┬───────────────┘
                │
      ┌─────────┴─────────┐
      │                   │
      ▼                   ▼
┌──────────────┐   ┌────────────────────┐
│ Baseline     │   │ Baseline + Overlap │
│ (no PP)      │   │ (e.g. 50%)         │
└──────┬───────┘   └──────────┬─────────┘
       │                      │
       ▼                      ▼
┌───────────────────────────────┐
│ Load signal per file          │
│ ACC+GYRO magnitudes           │
│ Windowing (step=window) OR    │
│ Windowing (step<window)       │
│ Feature extraction  +         │
│ inherit identity from registry│
└───────────────┬───────────────┘
                │
                ▼
      ┌─────────────────────────────┐
      │ windows_identity_baseline   │
      └───────────────┬─────────────┘
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
┌────────────────────┐   ┌──────────────────────┐
│ DP AFTER (features) │   │ DP BEFORE (signal)   │
│ Laplace on features │   │ Laplace on windows   │
│ eps=2,1,0.5         │   │ eps=2,1,0.5          │
└──────────┬─────────┘   └──────────┬───────────┘
           │                        │
           ▼                        ▼
  ┌─────────────────┐      ┌─────────────────┐
  │ CSV per epsilon │      │ CSV per epsilon │
  └────────┬────────┘      └────────┬────────┘
           │                        │
           └────────────┬───────────┘
                        ▼
           ┌──────────────────────────┐
           │ Modeling & Evaluation     │
           │ same subject split        │
           │ compare baseline vs PP    │
           └──────────────────────────┘
