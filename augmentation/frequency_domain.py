import numpy as np
from itertools import combinations

def make_pairs_same_condition_diff_severity(registers):
    # Group registers by condition
    groups = {}
    for r in registers:
        cond = r["condition"]
        groups.setdefault(cond, []).append(r)

    pairs = []

    # For each condition group, build pairs with different severity
    for cond, items in groups.items():
        for a, b in combinations(items, 2):
            if a["severity"] != b["severity"]:
                pairs.append((a, b))

    return pairs


def freq_augment(sample1, sample2):
    # Convert to numpy arrays
    x1 = np.asarray(sample1)
    x2 = np.asarray(sample2)

    # Ensure same length
    n = min(len(x1), len(x2))
    x1 = x1[:n]
    x2 = x2[:n]

    # FFT to frequency domain
    X1 = np.fft.fft(x1)
    X2 = np.fft.fft(x2)

    # Random mixing weight
    alpha = np.random.rand()

    # Random superposition in frequency domain
    X_mix = alpha * X1 + (1 - alpha) * X2

    # Back to time domain
    x_mix = np.fft.ifft(X_mix).real

    return x_mix

def augment_segments_mode3(segments_a, segments_b):
    # Use the smaller set size
    n = min(len(segments_a), len(segments_b))
    if n == 0:
        return np.empty((0, 0, 1))

    # Infer segment length from input
    seg_len = segments_a.shape[1]

    X_aug = np.empty((n, seg_len, 1), dtype=np.float32)

    for i in range(n):
        # Flatten to 1D
        x1 = segments_a[i, :, 0]
        x2 = segments_b[i, :, 0]

        # Mode 3 augmentation (frequency-domain random superposition)
        x_aug = freq_augment(x1, x2)

        # Store back as (seg_len, 1)
        X_aug[i, :, 0] = x_aug

    return X_aug