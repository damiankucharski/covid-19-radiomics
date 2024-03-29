import random
from enum import Enum

import pandas as pd
import numpy as np
from genetic_selection import GeneticSelectionCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mrmr import mrmr_classif
from sklearn.feature_selection import SelectorMixin

from src.preprocessing import DataframeTransformerWrapper as DTW


class SelectorType(Enum):
    LASSO = 'LASSO'
    GENETIC = 'GENETIC'
    SFS = 'SFS'
    PASSTHROUGH = 'PASSTHROUGH'
    MRMR = "MRMR"
    RF_TOP = "RF_TOP"


def get_selector_by_type(type: SelectorType):
    if type == SelectorType.LASSO:
        return LassoFeatureSelector
    if type == SelectorType.GENETIC:
        return GeneticFeatureSelector
    if type == SelectorType.SFS:
        return SequentialFeatureSelector
    if type == SelectorType.PASSTHROUGH:
        return PassthroughFeatureSelector
    if type == SelectorType.MRMR:
        return MRMRSelector
    if type == SelectorType.RF_TOP:
        return RandomForestTopFeatureSelector

    return None


class LassoFeatureSelector(SelectorMixin):
    def __init__(self, scoring="f1", cv=None, n_features=None):
        super(LassoFeatureSelector, self).__init__()
        self.best_C = None
        self.best_estimator = None
        self.scoring = scoring
        self.cv = cv
        self.n_features = n_features

    def fit(self, train_samples: pd.DataFrame, y=None):
        print("Selecting features using LASSO algorithm")
        self.expected_shape = train_samples.shape
        scaler = StandardScaler()
        train_samples = train_samples.copy()
        train_samples[:] = scaler.fit_transform(train_samples)

        np.random.seed(1992)


        lr = LogisticRegression(penalty="l1", random_state=1992, max_iter=100, solver='liblinear',
                                class_weight="balanced")
        gscv = GridSearchCV(lr, scoring=self.scoring, cv=self.cv, param_grid={"C": np.logspace(-4, 1, 50)}, verbose=2)
        gscv.fit(train_samples, y)

        self.best_C = gscv.best_params_["C"]
        self.best_estimator = gscv.best_estimator_
        print("The best C for LASSO:", self.best_C)

        self.coef = np.abs(self.best_estimator.coef_) if self.best_estimator.coef_.shape[0] == 1 else\
            np.abs(self.best_estimator.coef_).max(axis=0)[None, :]

        if self.n_features is None:
            self.support_ = ~(np.isclose(self.coef, [0], atol=1e-3)).squeeze()
        else:
            self.support_ = self.get_n_most_important_features(self.n_features, return_support=True)

        self.selected_features = np.array(train_samples.columns)[self.support_]

        return self

    def _get_support_mask(self):
        return self.support_

    def transform(self, X):
        return X.loc[:, self._get_support_mask()]

    def get_n_most_important_features(self, n_features, X = None, return_support=False):

        if X is None and not return_support:
            raise Exception("Either X or return_support must be provided")

        arange = np.arange(self.expected_shape[1])
        coef_argsort = self.coef.argsort()[::-1]  # descending
        support = np.zeros_like(arange, dtype=bool)
        support[coef_argsort[:self.n_features]] = True
        if return_support:
            return support
        else:
            return X.loc[:, support]


class GeneticFeatureSelector(SelectorMixin):
    def __init__(self, estimator=LogisticRegression(), scoring="f1", cv=None, max_features=20):
        super(GeneticFeatureSelector, self).__init__()
        self.estimator = estimator

        self.max_features = max_features
        self.best_estimator = None
        self.scoring = scoring
        self.cv = cv

    def fit(self, train_samples: pd.DataFrame, y=None):
        self.expected_shape = train_samples.shape
        np.random.seed(1992)
        random.seed(1992)
        self.selector = GeneticSelectionCV(self.estimator,
                                           cv=self.cv,
                                           verbose=1,
                                           scoring=self.scoring,
                                           max_features=self.max_features,
                                           n_population=300,
                                           crossover_proba=0.5,
                                           mutation_proba=0.2,
                                           n_generations=100,
                                           crossover_independent_proba=0.1,
                                           mutation_independent_proba=0.05,
                                           tournament_size=3,
                                           n_gen_no_change=20,
                                           caching=True,
                                           n_jobs=-1)
        print("Determining optimal features with genetic selector")
        self.selector.fit(train_samples, y)
        self.best_estimator = self.selector.estimator_
        self.selected_features = np.array(train_samples.columns[self.selector.support_])

        return self

    def _get_support_mask(self):
        return self.selector.support_

    def transform(self, X):
        return X.loc[:, self._get_support_mask()]


class SequentialFeatureSelector(SelectorMixin):
    def __init__(self, estimator = LogisticRegression(), scoring="f1", cv=None, forward=True, floating=False, k_features="best"):
        super(SequentialFeatureSelector, self).__init__()
        self.estimator = estimator
        self.best_estimator = None
        if isinstance(k_features, list):
            k_features = tuple(k_features)
        self.selector = SFS(self.estimator, k_features=k_features, scoring=scoring, cv=cv, n_jobs=-1,
                            verbose=2, forward=forward, floating=floating)

    def fit(self, train_samples: pd.DataFrame, y=None):
        print("Selecting features using SFS algorithm")
        self.expected_shape = train_samples.shape
        np.random.seed(1992)
        self.selector.fit(train_samples, y)
        features = np.asarray(self.selector.k_feature_idx_)
        self.support_ = np.zeros((self.expected_shape[1]), dtype=bool)
        self.support_[features] = True
        self.selected_features = np.array(train_samples.columns[list(features)])

        return self

    def _get_support_mask(self):
        return self.support_

    def transform(self, X):
        return X.loc[:, self._get_support_mask()]

class MRMRSelector(SelectorMixin):
    def __init__(self, K = 20, cv=None):
        super(MRMRSelector, self).__init__()
        self.selector = mrmr_classif
        self.K = K

    def fit(self, train_samples: pd.DataFrame, y=None):
        print("Selecting features using MRMR algorithm")
        self.expected_shape = train_samples.shape
        np.random.seed(1992)
        self.selected_features = self.selector(X=train_samples, y=y, K=self.K)
        self.support_ = np.zeros((self.expected_shape[1]), dtype=bool)
        for feature in self.selected_features:
            self.support_[train_samples.columns.get_loc(feature)] = True

        return self

    def _get_support_mask(self):
        return self.support_

    def transform(self, X):
        return X.loc[:, self._get_support_mask()]


class RandomForestTopFeatureSelector(SelectorMixin):

    def __init__(self, K=20, cv=None):
        self.K = K
        self.estimator = RandomForestClassifier(random_state=1992, verbose=1, n_jobs=-1)
        self.selector = None

    def fit(self, X, y):
        print("Selecting features with Random Forest importances")
        self.estimator.fit(X, y)
        self.selector = SelectFromModel(self.estimator, prefit=True, max_features=self.K)
        self.support_ = self.selector.get_support()
        return self

    def transform(self, X):
        return X.loc[:, self._get_support_mask()]

    def _get_support_mask(self):
        return self.support_

class PassthroughFeatureSelector(SelectorMixin):

    def __init__(self, cv=None):
        super(PassthroughFeatureSelector, self).__init__()

    def fit(self, train_samples: pd.DataFrame, y=None):
        self.expected_shape = train_samples.shape
        self.selected_features = train_samples.columns
        self.support_ = np.ones((self.expected_shape[1]), dtype=bool)

    def _get_support_mask(self):
        return self.support_

    def transform(self, X):
        return X.loc[:, self._get_support_mask()]
