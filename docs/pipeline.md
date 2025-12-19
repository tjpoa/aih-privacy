## Dataset handling

SisFall data are loaded through a dataset-specific loader that:
- parses subject and activity metadata from filenames
- converts raw sensor readings to physical units
- segments signals into fixed, non-overlapping 1s windows (200 samples)

This design allows extending the pipeline to other datasets with minimal changes.
