from dataclasses import dataclass
from pathlib import Path
from src.aih_privacy.config import DATA_RAW_DIR

@dataclass(frozen=True)
class DatasetSpec:
    name: str
    raw_dir: Path

DATASETS = {
    "sisfall": DatasetSpec(name="sisfall", raw_dir=DATA_RAW_DIR / "sisfall"),
}

def get_dataset(name: str) -> DatasetSpec:
    if name not in DATASETS:
        raise KeyError(f"Dataset '{name}' not registed. Options: {list(DATASETS)}")
    return DATASETS[name]
