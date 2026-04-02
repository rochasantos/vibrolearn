import numpy as np
import pywt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif


def energy_from_signal(signal, wavelet='db4', mode='symmetric', maxlevel=4):
    wp = pywt.WaveletPacket(data=signal, wavelet=wavelet, mode=mode, maxlevel=maxlevel)
    leaf_nodes = wp.get_leaf_nodes(True)
    energies = np.asarray(
        [np.sqrt(np.sum(np.asarray(node.data) ** 2)) / len(node.data) for node in leaf_nodes],
        dtype=np.float32
    )
    paths = [node.path for node in leaf_nodes]
    return paths, energies


def extract_features_with_paths(X, wavelet='db4', mode='symmetric', maxlevel=4):
    all_features = []
    all_paths = None

    for x in X:
        paths, energies = energy_from_signal(x, wavelet=wavelet, mode=mode, maxlevel=maxlevel)
        if all_paths is None:
            all_paths = paths
        all_features.append(energies)

    return all_paths, np.asarray(all_features, dtype=np.float32)


class WaveletPackage(TransformerMixin, BaseEstimator):
    def __init__(self, wavelet='db4', mode='symmetric', maxlevel=4):
        self.wavelet = wavelet
        self.mode = mode
        self.maxlevel = maxlevel
        self.feature_names_ = None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        n_channels = X.shape[-1]
        feature_blocks = []
        feature_names = []

        for ch in range(n_channels):
            paths, feats = extract_features_with_paths(
                X[:, :, ch],
                wavelet=self.wavelet,
                mode=self.mode,
                maxlevel=self.maxlevel
            )
            feature_blocks.append(feats)
            feature_names.extend([f"ch{ch}_{p}" for p in paths])

        self.feature_names_ = feature_names
        return np.concatenate(feature_blocks, axis=1)


class WPDFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, k=10, corr_threshold=0.9):
        self.k = k
        self.corr_threshold = corr_threshold
        self.selected_idx_ = None
        self.scores_ = None

    def fit(self, X, y):
        # Relevance
        scores = mutual_info_classif(X, y, random_state=0)
        self.scores_ = scores

        ranked_idx = np.argsort(scores)[::-1]

        # Redundancy removal
        selected = []
        for idx in ranked_idx:
            if len(selected) >= self.k:
                break

            keep = True
            for s in selected:
                corr = np.corrcoef(X[:, idx], X[:, s])[0, 1]
                if np.abs(corr) >= self.corr_threshold:
                    keep = False
                    break

            if keep:
                selected.append(idx)

        self.selected_idx_ = np.asarray(selected, dtype=int)
        return self

    def transform(self, X):
        return X[:, self.selected_idx_]