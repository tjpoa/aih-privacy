# Data directory

This project uses the SisFall dataset for fall detection experiments.

## Dataset
- Name: SisFall
- Sensors: Accelerometer and Gyroscope (IMU)
- Sampling rate: 200 Hz
- Subjects: young adults (SA) and elderly (SE)

## Local structure
The dataset must be placed locally as:

data/
└─ raw/
   └─ sisfall/
      ├─ D01_SA01_R01.txt
      ├─ F01_SE01_R01.txt
      └─ ...

## Notes on privacy
Raw data are not committed to this repository due to:
- dataset licensing constraints
- the presence of personal and quasi-identifiers
- GDPR compliance considerations
