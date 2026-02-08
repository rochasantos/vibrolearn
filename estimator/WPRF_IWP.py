from sklearn.base import BaseEstimator, ClassifierMixin
from feature.WaveletPackageFeatures import WaveletPackageFeatures as FeatureExtractor
from sklearn.ensemble import RandomForestClassifier as Estimator
from sklearn.ensemble import RandomForestClassifier as RFEstimator
from feature.WIFDFeatures import WIFDFeatures, WaveletPackagePlusWIFD
from feature.WaveletPackageFeatures import WaveletPackageFeatures


class WPRF(BaseEstimator, ClassifierMixin):
    """
    Drop-in model wrapper.

    IMPORTANT:
    - This class now supports `Estimator()` with no arguments, just like your current main.py.
    - You can still override fs/wavelet_params when you want.
    """

    def __init__(
        self,
        # Defaults to avoid breaking current code
        fs: float = 12000.0,
        wavelet_params: dict | None = None,
        use_wifd: bool = True,
        wifd_top_k_peaks: int = 10,
        wifd_min_freq: float = 0.0,
        wifd_max_freq: float | None = None,
        wifd_band_edges: list[tuple[float, float]] | None = None,
        estimator=None,
    ):
        if wavelet_params is None:
            # Safe default; adjust to match your WaveletPackage signature/choices
            wavelet_params = {"wavelet": "bior3.5", "maxlevel": 4}

        if estimator is None:
            estimator = RFEstimator()

        self.model = WaveletPackagePlusWIFD(
            estimator=estimator,
            wavelet_params=wavelet_params,
            fs=fs,
            use_wifd=use_wifd,
            wifd_top_k_peaks=wifd_top_k_peaks,
            wifd_min_freq=wifd_min_freq,
            wifd_max_freq=wifd_max_freq,
            wifd_band_edges=wifd_band_edges,
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)