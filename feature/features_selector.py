import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance


class BaseFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Base selector with common transform and support methods.
    """

    def __init__(self):
        self.selected_idx_ = None
        self.scores_ = None

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        return X[:, self.selected_idx_]

    def get_support(self):
        return self.selected_idx_


class WPDFeatureSelectorMI(BaseFeatureSelector):
    """
    Select top-k features using Mutual Information.
    """

    def __init__(self, k=12, random_state=0):
        super().__init__()
        self.k = k
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        scores = mutual_info_classif(X, y, random_state=self.random_state)
        self.scores_ = scores
        ranked_idx = np.argsort(scores)[::-1]
        self.selected_idx_ = ranked_idx[:self.k]
        return self


class WPDFeatureSelectorANOVA(BaseFeatureSelector):
    """
    Select top-k features using ANOVA F-score.
    """

    def __init__(self, k=12):
        super().__init__()
        self.k = k

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        scores, _ = f_classif(X, y)
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        self.scores_ = scores
        ranked_idx = np.argsort(scores)[::-1]
        self.selected_idx_ = ranked_idx[:self.k]
        return self


class WPDFeatureSelectorVariance(BaseFeatureSelector):
    """
    Select top-k features with highest variance.
    Useful as a very simple baseline.
    """

    def __init__(self, k=12):
        super().__init__()
        self.k = k

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float32)
        scores = np.var(X, axis=0)
        self.scores_ = scores
        ranked_idx = np.argsort(scores)[::-1]
        self.selected_idx_ = ranked_idx[:self.k]
        return self


class WPDFeatureSelectorRFImportance(BaseFeatureSelector):
    """
    Select top-k features using Random Forest feature importance.
    """

    def __init__(self, k=12, n_estimators=200, random_state=0):
        super().__init__()
        self.k = k
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)

        model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )
        model.fit(X, y)

        scores = model.feature_importances_
        self.scores_ = scores
        self.model_ = model

        ranked_idx = np.argsort(scores)[::-1]
        self.selected_idx_ = ranked_idx[:self.k]
        return self


class WPDFeatureSelectorPermutation(BaseFeatureSelector):
    """
    Select top-k features using permutation importance after fitting a model.
    This is slower, but often more reliable than raw impurity importance.
    """

    def __init__(self, k=12, estimator=None, n_repeats=10, random_state=0, scoring=None):
        super().__init__()
        self.k = k
        self.estimator = estimator
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.scoring = scoring
        self.model_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)

        if self.estimator is None:
            estimator = RandomForestClassifier(
                n_estimators=200,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            estimator = clone(self.estimator)

        estimator.fit(X, y)

        result = permutation_importance(
            estimator,
            X,
            y,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
            scoring=self.scoring,
            n_jobs=-1
        )

        scores = result.importances_mean
        self.scores_ = scores
        self.model_ = estimator

        ranked_idx = np.argsort(scores)[::-1]
        self.selected_idx_ = ranked_idx[:self.k]
        return self


class WPDFeatureSelectorCorrMI(BaseFeatureSelector):
    """
    Select features using Mutual Information, then remove highly correlated ones.
    Good balance between relevance and redundancy reduction.
    """

    def __init__(self, k=12, corr_threshold=0.9, random_state=0):
        super().__init__()
        self.k = k
        self.corr_threshold = corr_threshold
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)

        scores = mutual_info_classif(X, y, random_state=self.random_state)
        self.scores_ = scores
        ranked_idx = np.argsort(scores)[::-1]

        selected = []

        for idx in ranked_idx:
            if len(selected) >= self.k:
                break

            keep = True
            for s in selected:
                corr = np.corrcoef(X[:, idx], X[:, s])[0, 1]
                if np.isnan(corr):
                    corr = 0.0
                if abs(corr) >= self.corr_threshold:
                    keep = False
                    break

            if keep:
                selected.append(idx)

        self.selected_idx_ = np.asarray(selected, dtype=int)
        return self


class WPDFeatureSelectorCorrANOVA(BaseFeatureSelector):
    """
    Select features using ANOVA, then remove highly correlated ones.
    """

    def __init__(self, k=12, corr_threshold=0.9):
        super().__init__()
        self.k = k
        self.corr_threshold = corr_threshold

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)

        scores, _ = f_classif(X, y)
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        self.scores_ = scores
        ranked_idx = np.argsort(scores)[::-1]

        selected = []

        for idx in ranked_idx:
            if len(selected) >= self.k:
                break

            keep = True
            for s in selected:
                corr = np.corrcoef(X[:, idx], X[:, s])[0, 1]
                if np.isnan(corr):
                    corr = 0.0
                if abs(corr) >= self.corr_threshold:
                    keep = False
                    break

            if keep:
                selected.append(idx)

        self.selected_idx_ = np.asarray(selected, dtype=int)
        return self


class WPDFeatureSelectorRFStability(BaseFeatureSelector):
    """
    Select features using mean importance across individual trees.
    Also stores std of importance as a simple stability indicator.
    """

    def __init__(self, k=12, n_estimators=200, random_state=0):
        super().__init__()
        self.k = k
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.importance_std_ = None
        self.model_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)

        model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )
        model.fit(X, y)

        all_tree_importances = np.array(
            [tree.feature_importances_ for tree in model.estimators_],
            dtype=np.float32
        )

        mean_scores = np.mean(all_tree_importances, axis=0)
        std_scores = np.std(all_tree_importances, axis=0)

        self.scores_ = mean_scores
        self.importance_std_ = std_scores
        self.model_ = model

        ranked_idx = np.argsort(mean_scores)[::-1]
        self.selected_idx_ = ranked_idx[:self.k]
        return self