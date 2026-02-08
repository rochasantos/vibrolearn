import numpy as np
from itertools import combinations

def make_pairs_same_condition_same_load(registers):
    # Group registers by (condition, load)
    groups = {}
    for r in registers:
        key = (r["condition"], r["load"])
        groups.setdefault(key, []).append(r)

    pairs = []

    # Build all pairs inside each group
    for items in groups.values():
        pairs.extend(list(combinations(items, 2)))

    return pairs


def augment_time_mix_blocks(sample1, sample2, block_size=256):
    # Convert to 1D numpy arrays
    x1 = np.asarray(sample1).ravel()
    x2 = np.asarray(sample2).ravel()

    # Use the common length and keep only full blocks
    n = min(len(x1), len(x2))
    n = (n // block_size) * block_size
    if n == 0:
        empty = np.empty((0,), dtype=x1.dtype)
        return empty, empty

    x1 = x1[:n]
    x2 = x2[:n]

    # Split into blocks
    b1 = x1.reshape(-1, block_size)
    b2 = x2.reshape(-1, block_size)

    # Random choice per block
    mask = (np.random.rand(b1.shape[0]) < 0.5)[:, None]

    # Two complementary mixes: swapped sources where mask is False
    out1 = np.where(mask, b1, b2)
    out2 = np.where(mask, b2, b1)

    # Back to 1D signals
    return out1.reshape(-1), out2.reshape(-1)


def augment_segment_dam(segment, M=10, q=0.10, nmin=0.4, nmax=1.6, snr_db=20, rng=None):
    """
    Apply one of the 7 DAM strategies (DA1..DA7) with equal probability.
    Input/Output: 1D numpy array with the same length.
    """
    x = np.asarray(segment).reshape(-1).astype(float, copy=True)
    N = len(x)
    if N < 2:
        return x

    rng = np.random.default_rng() if rng is None else rng

    def _split_M(arr, M):
        # Split into M nearly-equal chunks (keeps order)
        idx = np.linspace(0, len(arr), M + 1).astype(int)
        return [arr[idx[i]:idx[i+1]] for i in range(M)]

    def _add_awgn(arr, snr_db):
        # Add white Gaussian noise at a target SNR (dB)
        p_signal = np.mean(arr ** 2) + 1e-12
        snr = 10 ** (snr_db / 10.0)
        p_noise = p_signal / snr
        noise = rng.normal(0.0, np.sqrt(p_noise), size=arr.shape)
        return arr + noise

    def DA1_local_reversing(arr):
        # Reverse a small window inside each of M chunks
        chunks = _split_M(arr, max(2, M))
        out = []
        for c in chunks:
            c = c.copy()
            Lc = len(c)
            if Lc < 4:
                out.append(c)
                continue
            win = max(2, int(q * Lc))
            start = rng.integers(0, Lc - win + 1)
            c[start:start+win] = c[start:start+win][::-1]
            out.append(c)
        return np.concatenate(out)

    def DA2_local_random_reversing(arr):
        # Reverse one random local segment of length floor(q*N)
        win = max(2, int(q * N))
        start = rng.integers(0, N - win + 1)
        out = arr.copy()
        out[start:start+win] = out[start:start+win][::-1]
        return out

    def DA3_global_reversing(arr):
        return arr[::-1].copy()

    def DA4_local_zooming(arr):
        # Scale (zoom amplitude) of local windows and insert back at 3 random non-overlapping positions
        out = arr.copy()
        win = max(2, int(q * N))
        if 3 * win >= N:
            # Fallback if signal too short
            factor = rng.random() * (nmax - nmin) + nmin
            start = rng.integers(0, N - win + 1)
            out[start:start+win] *= factor
            return out

        starts = []
        tries = 0
        while len(starts) < 3 and tries < 200:
            s = int(rng.integers(0, N - win + 1))
            if all(abs(s - t) >= win for t in starts):
                starts.append(s)
            tries += 1

        for s in starts:
            factor = rng.random() * (nmax - nmin) + nmin
            out[s:s+win] *= factor
        return out

    def DA5_global_zooming(arr):
        factor = rng.random() * (nmax - nmin) + nmin
        return arr * factor

    def DA6_local_segment_splicing(arr):
        # Split into M segments, shuffle their order, then splice back
        chunks = _split_M(arr, max(2, M))
        rng.shuffle(chunks)
        return np.concatenate(chunks)

    def DA7_noise_addition(arr):
        return _add_awgn(arr, snr_db=snr_db)

    strategies = [
        DA1_local_reversing,
        DA2_local_random_reversing,
        DA3_global_reversing,
        DA4_local_zooming,
        DA5_global_zooming,
        DA6_local_segment_splicing,
        DA7_noise_addition,
    ]

    f = strategies[int(rng.integers(0, len(strategies)))]  # equal probability
    y = f(x)

    # Safety: keep length exactly the same
    if len(y) != N:
        y = y[:N] if len(y) > N else np.pad(y, (0, N - len(y)), mode="edge")
    return y
