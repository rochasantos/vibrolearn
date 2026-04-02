# dataset/uored/experimenter.py

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from utils.assesment import holdout
from dataset.utils import (
    read_registers_from_config,
    split_acquisition,
    target_array,
)

segment_length = 1024


# ============================================================
# UORED domain split from Table 3
# ============================================================

UORED_DOMAIN_TABLE = {
    "1":  ["H-1-0",  "I-1-1",  "O-6-1",  "B-11-1", "C-16-1"],
    "2":  ["H-2-0",  "I-1-2",  "O-6-2",  "B-11-2", "C-16-2"],
    "3":  ["H-3-0",  "I-2-1",  "O-7-1",  "B-12-1", "C-17-1"],
    "4":  ["H-4-0",  "I-2-2",  "O-7-2",  "B-12-2", "C-17-2"],
    "5":  ["H-5-0",  "I-3-1",  "O-8-1",  "B-13-1", "C-18-1"],
    "6":  ["H-6-0",  "I-3-2",  "O-8-2",  "B-13-2", "C-18-2"],
    "7":  ["H-7-0",  "I-4-1",  "O-9-1",  "B-14-1", "C-19-1"],
    "8":  ["H-8-0",  "I-4-2",  "O-9-2",  "B-14-2", "C-19-2"],
    "9":  ["H-9-0",  "I-5-1",  "O-10-1", "B-15-1", "C-20-1"],
    "10": ["H-10-0", "I-5-2",  "O-10-2", "B-15-2", "C-20-2"],
}

# Train/validation: domains 2, 4, 6, 8
# Test: domain 10
papers_split = [
    ["2", "4", "6", "8"],
    ["10"],
]

# Allowed augmentation pairs
ALLOWED_AUGMENTATION_DOMAIN_PAIRS = {
    ("2", "6"),
    ("4", "8"),
}


# ============================================================
# UORED channels
# Matrix shape: (420000, 5)
#   0 vibration
#   1 acoustic
#   2 speed
#   3 load
#   4 temperature
# ============================================================

CHANNEL_INDEX = {
    "vibration": 0,
    "acoustic": 1,
    "speed": 2,
    "load": 3,
    "temperature": 4,
}


# ============================================================
# Helpers
# ============================================================

def _filename_stem_with_underscore(register: dict) -> str:
    """
    Example:
        B_11_1.mat -> B_11_1
    """
    return Path(str(register["filename"])).stem


def _filename_stem_with_hyphen(register: dict) -> str:
    """
    Example:
        B_11_1.mat -> B-11-1
    """
    return _filename_stem_with_underscore(register).replace("_", "-")


def _resolve_uored_file_path(raw_dir_path: str, register: dict) -> str:
    file_path = Path(raw_dir_path) / str(register["filename"])
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return str(file_path)


def _extract_domain_from_register(register: dict) -> str:
    """
    Domain is inferred from the sample_id / filename stem.

    Examples:
        H_2_0  -> domain 2
        I_1_2  -> domain 2
        O_8_2  -> domain 6
        B_12_2 -> domain 4
    """
    stem = _filename_stem_with_hyphen(register)

    for domain_id, members in UORED_DOMAIN_TABLE.items():
        if stem in members:
            return domain_id

    raise ValueError(f"Could not determine domain for register: {register['filename']}")


def _normalize_domain_pair(domain_a: str, domain_b: str) -> tuple[str, str]:
    return tuple(sorted((str(domain_a), str(domain_b))))


# ============================================================
# UORED loader
# ============================================================

def load_uored_acquisition(file_path: str, channels_columns: list[str]) -> np.ndarray:
    """
    Load one UORED acquisition from a .mat file.

    The data matrix is stored under a key equal to the file stem.
    Example:
        file_path: raw_data/uored/B_11_1.mat
        key:       B_11_1
    """
    mat = loadmat(file_path)
    stem = Path(file_path).stem

    if stem not in mat:
        available_keys = [k for k in mat.keys() if not k.startswith("__")]
        raise KeyError(
            f"Expected key '{stem}' not found in file '{file_path}'. "
            f"Available keys: {available_keys}"
        )

    data = mat[stem]

    if not isinstance(data, np.ndarray):
        raise ValueError(f"Key '{stem}' in '{file_path}' is not a numpy array.")

    if data.ndim != 2:
        raise ValueError(
            f"Expected 2D array in key '{stem}' from '{file_path}', got shape {data.shape}"
        )

    if channels_columns is None or len(channels_columns) == 0:
        return data.astype(np.float32)

    indices = [CHANNEL_INDEX[ch] for ch in channels_columns]
    return data[:, indices].astype(np.float32)


def get_uored_acquisition_data(
    raw_dir_path: str,
    channels_columns: list[str],
    load_acquisition_func,
    register: dict,
) -> np.ndarray:
    file_path = _resolve_uored_file_path(raw_dir_path, register)
    return load_acquisition_func(file_path, channels_columns)


def extract_uored_segments_and_targets(
    raw_dir_path: str,
    channels_columns: list[str],
    segment_length: int,
    load_acquisition_func,
    register: dict,
):
    """
    Output shape forced to (N, C, L).
    """
    acquisition = get_uored_acquisition_data(
        raw_dir_path=raw_dir_path,
        channels_columns=channels_columns,
        load_acquisition_func=load_acquisition_func,
        register=register,
    )

    segs = split_acquisition(acquisition[:4096], segment_length=segment_length)

    if segs.ndim == 2:
        segs = segs[:, np.newaxis, :]
    elif segs.ndim == 3:
        segs = np.transpose(segs, (0, 2, 1))
    else:
        raise ValueError(f"Unexpected segmented shape: {segs.shape}")

    targets = target_array(str(register["condition"]), segs.shape[0])
    return segs.astype(np.float32), targets


# ============================================================
# Mode 3 augmentation at acquisition level
# NEW RULE:
#   same class
#   only domain 2 with 6
#   only domain 4 with 8
# ============================================================

def _rfft_acquisition(acq: np.ndarray) -> np.ndarray:
    return np.fft.rfft(acq, axis=0)


def _irfft_acquisition(Xf: np.ndarray, n_time: int) -> np.ndarray:
    x = np.fft.irfft(Xf, n=n_time, axis=0)
    return x.astype(np.float32)


def _align_acquisitions(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    T = min(a.shape[0], b.shape[0])
    return a[:T], b[:T], T


def _mix_two_acquisitions(acq_a: np.ndarray, acq_b: np.ndarray) -> np.ndarray:
    acq_a, acq_b, T = _align_acquisitions(acq_a, acq_b)
    Xf_a = _rfft_acquisition(acq_a)
    Xf_b = _rfft_acquisition(acq_b)
    Xf_mix = Xf_a + Xf_b
    acq_mix = _irfft_acquisition(Xf_mix, n_time=T)
    return acq_mix


def _build_augmented_segments_from_registers(
    registers: list[dict],
    raw_dir_path: str,
    channels_columns: list[str],
    segment_length: int,
    load_acquisition_func,
    rng: np.random.Generator,
    mixes_per_pair: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build augmented data with the rule:
      - same class (condition)
      - domain 2 mixed only with domain 6
      - domain 4 mixed only with domain 8

    Groups:
        groups[condition][domain] = [(file_id, acquisition), ...]
    """
    groups: dict[str, dict[str, list[tuple[str, np.ndarray]]]] = {}

    for reg in registers:
        cond = str(reg.get("condition"))
        domain = _extract_domain_from_register(reg)

        # Only training domains participating in augmentation
        if domain not in {"2", "4", "6", "8"}:
            continue

        acq = get_uored_acquisition_data(
            raw_dir_path,
            channels_columns,
            load_acquisition_func,
            reg,
        ).astype(np.float32)

        file_id = str(reg["filename"])
        groups.setdefault(cond, {}).setdefault(domain, []).append((file_id, acq))

    X_aug_list: list[np.ndarray] = []
    y_aug_list: list[np.ndarray] = []

    for cond, domain_dict in groups.items():
        for domain_a, domain_b in ALLOWED_AUGMENTATION_DOMAIN_PAIRS:
            list_a = domain_dict.get(domain_a, [])
            list_b = domain_dict.get(domain_b, [])

            if not list_a or not list_b:
                continue

            n_base = min(len(list_a), len(list_b))
            n_new = mixes_per_pair * n_base

            if n_new <= 0:
                continue

            idx_a = rng.integers(0, len(list_a), size=n_new)
            idx_b = rng.integers(0, len(list_b), size=n_new)

            for ia, ib in zip(idx_a, idx_b):
                file_a, acq_a = list_a[ia]
                file_b, acq_b = list_b[ib]

                if file_a == file_b:
                    continue

                acq_mix = _mix_two_acquisitions(acq_a, acq_b)

                segs = split_acquisition(acq_mix, segment_length=segment_length)
                if segs.shape[0] == 0:
                    continue

                # Convert to (N, C, L)
                if segs.ndim == 2:
                    segs = segs[:, np.newaxis, :]
                elif segs.ndim == 3:
                    segs = np.transpose(segs, (0, 2, 1))
                else:
                    raise ValueError(
                        f"Unexpected augmented segmented shape: {segs.shape}"
                    )

                X_aug_list.append(segs.astype(np.float32))
                y_aug_list.append(target_array(cond, segs.shape[0]))

    if not X_aug_list:
        n_channels = len(channels_columns) if channels_columns else 1
        return (
            np.empty((0, n_channels, segment_length), dtype=np.float32),
            np.empty((0,), dtype="U20"),
        )

    X_aug = np.concatenate(X_aug_list, axis=0)
    y_aug = np.concatenate(y_aug_list, axis=0)
    return X_aug, y_aug


def get_X_y_augmented_acquisition_level(
    registers,
    raw_dir_path,
    channels_columns,
    segment_length,
    load_acquisition_func,
    rng: np.random.Generator,
    mixes_per_pair: int = 1,
    augment: bool = True,
):
    X_list = []
    y_list = []

    if len(registers) > 0:
        for reg in registers:
            segs, targets = extract_uored_segments_and_targets(
                raw_dir_path,
                channels_columns,
                segment_length,
                load_acquisition_func,
                reg,
            )
            X_list.append(segs.astype(np.float32))
            y_list.append(targets)

    if not X_list:
        n_channels = len(channels_columns) if channels_columns else 1
        X_orig = np.empty((0, n_channels, segment_length), dtype=np.float32)
        y_orig = np.empty((0,), dtype="U20")
    else:
        X_orig = np.concatenate(X_list, axis=0)
        y_orig = np.concatenate(y_list, axis=0)

    if not augment:
        return X_orig, y_orig

    X_aug, y_aug = _build_augmented_segments_from_registers(
        registers=registers,
        raw_dir_path=raw_dir_path,
        channels_columns=channels_columns,
        segment_length=segment_length,
        load_acquisition_func=load_acquisition_func,
        rng=rng,
        mixes_per_pair=mixes_per_pair,
    )

    X_out = np.concatenate([X_orig, X_aug], axis=0)
    y_out = np.concatenate([y_orig, y_aug], axis=0)
    return X_out, y_out


def get_list_of_X_y_mode3(
    list_of_folds,
    raw_dir_path,
    channels_columns,
    segment_length,
    load_acquisition_func,
    seed: int = 0,
    mixes_per_pair: int = 1,
    test_fold_index: int | None = None,
    augment: bool = False,
):
    rng = np.random.default_rng(seed)
    list_of_X_y = []

    for fold_idx, fold in enumerate(list_of_folds):
        if test_fold_index is not None and fold_idx == test_fold_index:
            X_list = []
            y_list = []

            if len(fold) == 0:
                n_channels = len(channels_columns) if channels_columns else 1
                X = np.empty((0, n_channels, segment_length), dtype=np.float32)
                y = np.empty((0,), dtype="U20")
            else:
                for reg in fold:
                    segs, targets = extract_uored_segments_and_targets(
                        raw_dir_path,
                        channels_columns,
                        segment_length,
                        load_acquisition_func,
                        reg,
                    )
                    X_list.append(segs.astype(np.float32))
                    y_list.append(targets)

                n_channels = len(channels_columns) if channels_columns else 1
                X = (
                    np.concatenate(X_list, axis=0)
                    if X_list else np.empty((0, n_channels, segment_length), dtype=np.float32)
                )
                y = (
                    np.concatenate(y_list, axis=0)
                    if y_list else np.empty((0,), dtype="U20")
                )

            list_of_X_y.append((X, y))
            continue

        X, y = get_X_y_augmented_acquisition_level(
            registers=fold,
            raw_dir_path=raw_dir_path,
            channels_columns=channels_columns,
            segment_length=segment_length,
            load_acquisition_func=load_acquisition_func,
            rng=rng,
            mixes_per_pair=mixes_per_pair,
            augment=augment,
        )
        list_of_X_y.append((X, y))

    return list_of_X_y


# ============================================================
# Fold construction
# ============================================================

def _register_belongs_to_domain(register: dict, domain_ids: list[str]) -> bool:
    stem = _filename_stem_with_hyphen(register)

    for domain_id in domain_ids:
        if stem in UORED_DOMAIN_TABLE[str(domain_id)]:
            return True

    return False


def get_papers_split(domain_ids: list[str]):
    config_file = "dataset/uored/config.csv"
    registers = read_registers_from_config(config_file)
    filtered = [reg for reg in registers if _register_belongs_to_domain(reg, domain_ids)]
    return filtered


def get_list_of_papers_splits():
    folds = []
    for domain_ids in papers_split:
        folds.append(get_papers_split(domain_ids))
    return folds


# ============================================================
# Public API
# ============================================================

def run_papers_experiment(
    model,
    list_of_metrics,
    mixes_per_pair: int = 1,
    seed: int = 0,
    augment: bool = False,
    channel: str = "vibration",
):
    test_fold_index = 1
    list_of_folds = get_list_of_papers_splits()

    list_of_X_y = get_list_of_X_y_mode3(
        list_of_folds=list_of_folds,
        raw_dir_path="raw_data/uored",
        channels_columns=[channel],
        segment_length=segment_length,
        load_acquisition_func=load_uored_acquisition,
        mixes_per_pair=mixes_per_pair,
        seed=seed,
        test_fold_index=test_fold_index,
        augment=augment,
    )

    scores = holdout(
        model,
        list_of_X_y,
        test_fold_index=test_fold_index,
        list_of_metrics=list_of_metrics,
    )
    return scores