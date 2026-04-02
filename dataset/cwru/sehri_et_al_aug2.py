import numpy as np

from utils.assesment import holdout
from dataset.utils import (
    read_registers_from_config,
    filter_registers_by_key_value_sequence,
    load_matlab_acquisition,
    extract_segments_and_targets,   # original segments
    get_acquisition_data,           # full acquisition
    split_acquisition,              # split acquisition into segments
    target_array,                   # create label array
)

segment_length = 1024

#            ### training -------------------------###   ### testing --###
# papers_split = [(['0', '1', '2', '3'], ['0.007', '0.014']), (['0'], ['0.021'])]
# papers_split = [(['0', '1', '2', '3'], ['0.007', '0.014']), (['0', '1', '2', '3'], ['0.021'])]
papers_split = [(['0', '1', '2', '3'], ['0.007', '0.021']), (['0', '1', '2', '3'], ['0.014'])]
# papers_split = [(['0', '1', '2', '3'], ['0.014', '0.021']), (['0', '1', '2', '3'], ['0.007'])]


# ---------------------------------------------------------------------
# MODE 3 AUGMENTATION (ACQUISITION LEVEL)
# ---------------------------------------------------------------------

def _rfft_acquisition(acq: np.ndarray) -> np.ndarray:
    """
    acq: (T, C)
    returns: (F, C)
    """
    return np.fft.rfft(acq, axis=0)


def _irfft_acquisition(Xf: np.ndarray, n_time: int) -> np.ndarray:
    """
    Xf: (F, C)
    returns: (T, C)
    """
    x = np.fft.irfft(Xf, n=n_time, axis=0)
    return x.astype(np.float32)


def _align_acquisitions(a: np.ndarray, b: np.ndarray):
    """
    Trim both acquisitions to the same minimum length.
    """
    T = min(a.shape[0], b.shape[0])
    return a[:T], b[:T], T


def _mode3_mix_two_acquisitions(acq_a: np.ndarray, acq_b: np.ndarray) -> np.ndarray:
    """
    Full-spectrum frequency-domain superposition:
      FFT(A) + FFT(B) -> IFFT
    """
    acq_a, acq_b, T = _align_acquisitions(acq_a, acq_b)

    Xf_a = _rfft_acquisition(acq_a)
    Xf_b = _rfft_acquisition(acq_b)

    Xf_mix = Xf_a + Xf_b
    acq_mix = _irfft_acquisition(Xf_mix, n_time=T)
    return acq_mix


def _build_mode3_segments_same_load_diff_severity(
    registers: list[dict],
    raw_dir_path: str,
    channels_columns: list[str],
    segment_length: int,
    load_acquisition_func,
    rng: np.random.Generator,
    mixes_per_pair: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Same load + same condition, but different severities.
    """
    groups: dict[tuple[str, str], dict[str, list[np.ndarray]]] = {}

    for reg in registers:
        ld = str(reg.get("load"))
        cond = str(reg.get("condition"))
        sev = str(reg.get("severity"))

        acq = get_acquisition_data(raw_dir_path, channels_columns, load_acquisition_func, reg)
        acq = acq.astype(np.float32)

        key = (ld, cond)
        groups.setdefault(key, {}).setdefault(sev, []).append(acq)

    X_aug_list = []
    y_aug_list = []

    for (ld, cond), sev_dict in groups.items():
        sevs = sorted(sev_dict.keys())
        if len(sevs) < 2:
            continue

        for i in range(len(sevs)):
            for j in range(i + 1, len(sevs)):
                sev_a, sev_b = sevs[i], sevs[j]
                list_a = sev_dict.get(sev_a, [])
                list_b = sev_dict.get(sev_b, [])
                if not list_a or not list_b:
                    continue

                n_base = min(len(list_a), len(list_b))
                n_new = mixes_per_pair * n_base
                if n_new <= 0:
                    continue

                idx_a = rng.integers(0, len(list_a), size=n_new)
                idx_b = rng.integers(0, len(list_b), size=n_new)

                for ka, kb in zip(idx_a, idx_b):
                    acq_mix = _mode3_mix_two_acquisitions(list_a[ka], list_b[kb])
                    segs = split_acquisition(acq_mix, segment_length=segment_length)
                    if segs.shape[0] == 0:
                        continue

                    X_aug_list.append(segs.astype(np.float32))
                    y_aug_list.append(target_array(cond, segs.shape[0]))

    if not X_aug_list:
        return (
            np.empty((0, segment_length, 1), dtype=np.float32),
            np.empty((0,), dtype="U10"),
        )

    X_aug = np.concatenate(X_aug_list, axis=0)
    y_aug = np.concatenate(y_aug_list, axis=0)
    return X_aug, y_aug


def _build_mode3_segments_same_severity_diff_load(
    registers: list[dict],
    raw_dir_path: str,
    channels_columns: list[str],
    segment_length: int,
    load_acquisition_func,
    rng: np.random.Generator,
    mixes_per_pair: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Same severity + same condition, but different loads.
    """
    groups: dict[tuple[str, str], dict[str, list[np.ndarray]]] = {}

    for reg in registers:
        ld = str(reg.get("load"))
        cond = str(reg.get("condition"))
        sev = str(reg.get("severity"))

        acq = get_acquisition_data(raw_dir_path, channels_columns, load_acquisition_func, reg)
        acq = acq.astype(np.float32)

        key = (sev, cond)
        groups.setdefault(key, {}).setdefault(ld, []).append(acq)

    X_aug_list = []
    y_aug_list = []

    for (sev, cond), load_dict in groups.items():
        loads = sorted(load_dict.keys())
        if len(loads) < 2:
            continue

        for i in range(len(loads)):
            for j in range(i + 1, len(loads)):
                ld_a, ld_b = loads[i], loads[j]
                list_a = load_dict.get(ld_a, [])
                list_b = load_dict.get(ld_b, [])
                if not list_a or not list_b:
                    continue

                n_base = min(len(list_a), len(list_b))
                n_new = mixes_per_pair * n_base
                if n_new <= 0:
                    continue

                idx_a = rng.integers(0, len(list_a), size=n_new)
                idx_b = rng.integers(0, len(list_b), size=n_new)

                for ka, kb in zip(idx_a, idx_b):
                    acq_mix = _mode3_mix_two_acquisitions(list_a[ka], list_b[kb])
                    segs = split_acquisition(acq_mix, segment_length=segment_length)
                    if segs.shape[0] == 0:
                        continue

                    X_aug_list.append(segs.astype(np.float32))
                    y_aug_list.append(target_array(cond, segs.shape[0]))

    if not X_aug_list:
        return (
            np.empty((0, segment_length, 1), dtype=np.float32),
            np.empty((0,), dtype="U10"),
        )

    X_aug = np.concatenate(X_aug_list, axis=0)
    y_aug = np.concatenate(y_aug_list, axis=0)
    return X_aug, y_aug


# ---------------------------------------------------------------------
# DOMINANT SHUFFLE (SEGMENT LEVEL, INDIVIDUAL SIGNAL)
# ---------------------------------------------------------------------

def _dominant_shuffle_segment(
    seg: np.ndarray,
    rng: np.random.Generator,
    k: int = 20,
    p_shuffle: float = 1.0,
    ignore_dc: bool = True,
) -> np.ndarray:
    """
    seg: (L, C)
    returns: (L, C)

    Dominant Shuffle:
      1) FFT of each segment
      2) select top-k dominant bins by magnitude
      3) permute only these dominant bins
      4) IFFT back to time domain
    """
    if p_shuffle < 1.0 and rng.random() > p_shuffle:
        return seg

    L, C = seg.shape
    Xf = np.fft.rfft(seg, axis=0)
    F = Xf.shape[0]

    k_eff = int(min(max(k, 1), F))
    Xf_new = Xf.copy()

    for c in range(C):
        mag = np.abs(Xf[:, c])

        if ignore_dc and F > 0:
            mag = mag.copy()
            mag[0] = -np.inf

        idx = np.argpartition(mag, -k_eff)[-k_eff:]
        perm = rng.permutation(idx)
        Xf_new[idx, c] = Xf[perm, c]

    seg_aug = np.fft.irfft(Xf_new, n=L, axis=0).astype(np.float32)
    return seg_aug


def _build_dominant_shuffle_from_segments(
    X_orig: np.ndarray,
    y_orig: np.ndarray,
    rng: np.random.Generator,
    dominant_k: int = 20,
    p_shuffle: float = 1.0,
    ignore_dc: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create one dominant-shuffle augmented segment for each original segment.
    """
    if X_orig.shape[0] == 0:
        return (
            np.empty((0, X_orig.shape[1], X_orig.shape[2]), dtype=np.float32),
            np.empty((0,), dtype=y_orig.dtype),
        )

    X_aug = np.empty_like(X_orig, dtype=np.float32)

    for i in range(X_orig.shape[0]):
        X_aug[i] = _dominant_shuffle_segment(
            X_orig[i],
            rng=rng,
            k=dominant_k,
            p_shuffle=p_shuffle,
            ignore_dc=ignore_dc,
        )

    y_aug = y_orig.copy()
    return X_aug, y_aug


# ---------------------------------------------------------------------
# ORIGINAL SEGMENTS
# ---------------------------------------------------------------------

def get_X_y_original_segments(
    registers,
    raw_dir_path,
    channels_columns,
    segment_length,
    load_acquisition_func,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build only original segments.
    """
    X_list = []
    y_list = []

    for reg in registers:
        segs, targets = extract_segments_and_targets(
            raw_dir_path, channels_columns, segment_length, load_acquisition_func, reg
        )
        X_list.append(segs.astype(np.float32))
        y_list.append(targets)

    if not X_list:
        return (
            np.empty((0, segment_length, 1), dtype=np.float32),
            np.empty((0,), dtype="U10"),
        )

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y


# ---------------------------------------------------------------------
# COMBINED DATASET: ORIGINAL + MODE3 + DOMINANT SHUFFLE
# ---------------------------------------------------------------------

def get_X_y_with_all_augmentations(
    registers,
    raw_dir_path,
    channels_columns,
    segment_length,
    load_acquisition_func,
    rng: np.random.Generator,
    mixes_per_pair: int = 1,
    augment: bool = True,
    dominant_k: int = 20,
    p_shuffle: float = 1.0,
    ignore_dc: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      - if augment=False:
          X = original only
      - if augment=True:
          X = original + mode3(same load diff severity)
                    + mode3(same severity diff load)
                    + dominant shuffle(segment-level)
    """
    # Original
    X_orig, y_orig = get_X_y_original_segments(
        registers=registers,
        raw_dir_path=raw_dir_path,
        channels_columns=channels_columns,
        segment_length=segment_length,
        load_acquisition_func=load_acquisition_func,
    )

    if not augment:
        return X_orig, y_orig

    # Mode 3 - same load + diff severity
    X_mode3_sev, y_mode3_sev = _build_mode3_segments_same_load_diff_severity(
        registers=registers,
        raw_dir_path=raw_dir_path,
        channels_columns=channels_columns,
        segment_length=segment_length,
        load_acquisition_func=load_acquisition_func,
        rng=rng,
        mixes_per_pair=mixes_per_pair,
    )

    # Mode 3 - same severity + diff load
    X_mode3_load, y_mode3_load = _build_mode3_segments_same_severity_diff_load(
        registers=registers,
        raw_dir_path=raw_dir_path,
        channels_columns=channels_columns,
        segment_length=segment_length,
        load_acquisition_func=load_acquisition_func,
        rng=rng,
        mixes_per_pair=mixes_per_pair,
    )

    # Dominant Shuffle from original segments
    X_dom, y_dom = _build_dominant_shuffle_from_segments(
        X_orig=X_orig,
        y_orig=y_orig,
        rng=rng,
        dominant_k=dominant_k,
        p_shuffle=p_shuffle,
        ignore_dc=ignore_dc,
    )

    X_parts = [X_orig]
    y_parts = [y_orig]

    if X_mode3_sev.shape[0] > 0:
        X_parts.append(X_mode3_sev)
        y_parts.append(y_mode3_sev)

    # if X_mode3_load.shape[0] > 0:
    #     X_parts.append(X_mode3_load)
    #     y_parts.append(y_mode3_load)

    # if X_dom.shape[0] > 0:
    #     X_parts.append(X_dom)
    #     y_parts.append(y_dom)

    X_out = np.concatenate(X_parts, axis=0)
    y_out = np.concatenate(y_parts, axis=0)
    return X_out, y_out


def get_list_of_X_y_by_fold(
    list_of_folds,
    raw_dir_path,
    channels_columns,
    segment_length,
    load_acquisition_func,
    seed: int = 0,
    mixes_per_pair: int = 1,
    test_fold_index: int | None = None,
    augment: bool = False,
    dominant_k: int = 20,
    p_shuffle: float = 1.0,
    ignore_dc: bool = True,
):
    """
    Returns list_of_X_y where:
      - training folds: original + both augmentations
      - test fold: original only
    """
    rng = np.random.default_rng(seed)
    list_of_X_y = []

    for fold_idx, fold in enumerate(list_of_folds):
        is_test = (test_fold_index is not None and fold_idx == test_fold_index)

        X, y = get_X_y_with_all_augmentations(
            registers=fold,
            raw_dir_path=raw_dir_path,
            channels_columns=channels_columns,
            segment_length=segment_length,
            load_acquisition_func=load_acquisition_func,
            rng=rng,
            mixes_per_pair=mixes_per_pair,
            augment=(augment and not is_test),
            dominant_k=dominant_k,
            p_shuffle=p_shuffle,
            ignore_dc=ignore_dc,
        )
        list_of_X_y.append((X, y))

    return list_of_X_y


# ---------------------------------------------------------------------
# FOLD CONSTRUCTION
# ---------------------------------------------------------------------

def get_papers_split(loads, severities):
    sample_rate = "48000"
    config_file = "dataset/cwru/config.csv"
    prlzs = ["None", "6"]

    registers = read_registers_from_config(config_file)
    filtered = filter_registers_by_key_value_sequence(
        registers,
        [
            ("sample_rate", [sample_rate]),
            ("load", loads),
            ("severity", severities),
            ("prlz", prlzs),
        ],
    )
    return filtered


def get_list_of_papers_splits():
    folds = []
    for loads, severities in papers_split:
        folds.append(get_papers_split(loads, severities))
    return folds


# ---------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------

def run_papers_experiment(
    model,
    list_of_metrics,
    mixes_per_pair: int = 1,
    seed: int = 0,
    augment: bool = False,
    dominant_k: int = 20,
    p_shuffle: float = 1.0,
    ignore_dc: bool = True,
):
    """
    Hold-out experiment:
      - training fold(s): original + Mode 3 + Dominant Shuffle
      - test fold: original only
    """
    test_fold_index = 1
    list_of_folds = get_list_of_papers_splits()

    list_of_X_y = get_list_of_X_y_by_fold(
        list_of_folds=list_of_folds,
        raw_dir_path="raw_data/cwru",
        channels_columns=["DE"],
        segment_length=segment_length,
        load_acquisition_func=load_matlab_acquisition,
        seed=seed,
        mixes_per_pair=mixes_per_pair,
        test_fold_index=test_fold_index,
        augment=augment,
        dominant_k=dominant_k,
        p_shuffle=p_shuffle,
        ignore_dc=ignore_dc,
    )

    scores = holdout(
        model,
        list_of_X_y,
        test_fold_index=test_fold_index,
        list_of_metrics=list_of_metrics,
    )
    return scores