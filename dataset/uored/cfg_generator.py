# dataset/uored/build_uored_config.py

from __future__ import annotations

import csv
from pathlib import Path


def _make_filename(sample_id: str, extension: str = ".mat") -> str:
    """
    Convert dataset ID from hyphen format to actual filename format.

    Example:
        H-2-0 -> H_2_0.mat
        I-1-2 -> I_1_2.mat
    """
    return sample_id.replace("-", "_") + extension


def build_uored_config(output_csv: str | Path) -> None:
    """
    Build a config CSV for the UORED-VAFCLS dataset.

    Assumptions:
    - Dataset IDs follow the article notation, e.g. H-2-0, I-1-2, O-10-2.
    - Actual local filenames use underscores instead of hyphens:
        H_2_0.mat, I_1_2.mat, O_10_2.mat, ...
    - Raw files are stored locally in raw_data/uored.
    """

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "dataset_name",
        "sample_id",
        "filename",
        "label",
        "condition",
        "fault_state",
        "bearing_id",
        "sampling_rate",
        "duration_seconds",
        "n_samples",
        "channels",
    ]

    rows: list[dict] = []

    # Healthy: H-1-0 ... H-20-0
    for bearing_id in range(1, 21):
        sample_id = f"H-{bearing_id}-0"
        rows.append(
            {
                "dataset_name": "UORED-VAFCLS",
                "sample_id": sample_id,
                "filename": _make_filename(sample_id),
                "label": "H",
                "condition": "Healthy",
                "fault_state": "Healthy",
                "bearing_id": str(bearing_id),
                "sampling_rate": "42000",
                "duration_seconds": "10",
                "n_samples": "420000",
                "channels": "vibration,acoustic,speed,load,temperature",
            }
        )

    # Inner Race: I-1-1 ... I-5-2
    for bearing_id in range(1, 6):
        for fault_state, state_name in [(1, "Developing"), (2, "Faulty")]:
            sample_id = f"I-{bearing_id}-{fault_state}"
            rows.append(
                {
                    "dataset_name": "UORED-VAFCLS",
                    "sample_id": sample_id,
                    "filename": _make_filename(sample_id),
                    "label": "I",
                    "condition": "Inner Race",
                    "fault_state": state_name,
                    "bearing_id": str(bearing_id),
                    "sampling_rate": "42000",
                    "duration_seconds": "10",
                    "n_samples": "420000",
                    "channels": "vibration,acoustic,speed,load,temperature",
                }
            )

    # Outer Race: O-6-1 ... O-10-2
    for bearing_id in range(6, 11):
        for fault_state, state_name in [(1, "Developing"), (2, "Faulty")]:
            sample_id = f"O-{bearing_id}-{fault_state}"
            rows.append(
                {
                    "dataset_name": "UORED-VAFCLS",
                    "sample_id": sample_id,
                    "filename": _make_filename(sample_id),
                    "label": "O",
                    "condition": "Outer Race",
                    "fault_state": state_name,
                    "bearing_id": str(bearing_id),
                    "sampling_rate": "42000",
                    "duration_seconds": "10",
                    "n_samples": "420000",
                    "channels": "vibration,acoustic,speed,load,temperature",
                }
            )

    # Ball: B-11-1 ... B-15-2
    for bearing_id in range(11, 16):
        for fault_state, state_name in [(1, "Developing"), (2, "Faulty")]:
            sample_id = f"B-{bearing_id}-{fault_state}"
            rows.append(
                {
                    "dataset_name": "UORED-VAFCLS",
                    "sample_id": sample_id,
                    "filename": _make_filename(sample_id),
                    "label": "B",
                    "condition": "Ball",
                    "fault_state": state_name,
                    "bearing_id": str(bearing_id),
                    "sampling_rate": "42000",
                    "duration_seconds": "10",
                    "n_samples": "420000",
                    "channels": "vibration,acoustic,speed,load,temperature",
                }
            )

    # Cage: C-16-1 ... C-20-2
    for bearing_id in range(16, 21):
        for fault_state, state_name in [(1, "Developing"), (2, "Faulty")]:
            sample_id = f"C-{bearing_id}-{fault_state}"
            rows.append(
                {
                    "dataset_name": "UORED-VAFCLS",
                    "sample_id": sample_id,
                    "filename": _make_filename(sample_id),
                    "label": "C",
                    "condition": "Cage",
                    "fault_state": state_name,
                    "bearing_id": str(bearing_id),
                    "sampling_rate": "42000",
                    "duration_seconds": "10",
                    "n_samples": "420000",
                    "channels": "vibration,acoustic,speed,load,temperature",
                }
            )

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"UORED config generated successfully at: {output_csv}")


if __name__ == "__main__":
    build_uored_config("dataset/uored/config.csv")