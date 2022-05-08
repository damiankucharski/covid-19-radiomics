import random
from enum import Enum

import pandas as pd
import numpy as np
from genetic_selection import GeneticSelectionCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

from sklearn.feature_selection import SelectorMixin


class SelectorType(Enum):
    LASSO = 'LASSO'
    GENETIC = 'GENETIC'
    SFS = 'SFS'
    PASSTHROUGH = 'PASSTHROUGH'


def get_selector_by_type(type: SelectorType):
    if type == SelectorType.LASSO:
        return LassoFeatureSelector
    if type == SelectorType.GENETIC:
        return GeneticFeatureSelector
    if type == SelectorType.SFS:
        return SequentialFeatureSelector
    if type == SelectorType.PASSTHROUGH:
        return PassthroughFeatureSelector

    return None


class LassoFeatureSelector(SelectorMixin):
    def __init__(self, scoring="f1", cv=None):
        super(LassoFeatureSelector, self).__init__()
        self.best_C = None
        self.best_estimator = None
        self.scoring = scoring
        self.cv = cv

    def fit(self, train_samples: pd.DataFrame, y=None):
        print("Selecting features using LASSO algorithm")
        self.expected_shape = train_samples.shape
        scaler = StandardScaler()
        train_samples = train_samples.copy()
        train_samples[:] = scaler.fit_transform(train_samples)

        np.random.seed(1992)
        lr = LogisticRegression(penalty="l1", random_state=1992, max_iter=1000, solver="liblinear",
                                class_weight="balanced")
        gscv = GridSearchCV(lr, scoring=self.scoring, cv=self.cv, param_grid={"C": np.logspace(-4, 4, 100)})
        gscv.fit(train_samples, y)

        self.best_C = gscv.best_params_["C"]
        self.best_estimator = gscv.best_estimator_
        print("The best C for LASSO:", self.best_C)

        where_mask = ~(np.isclose(self.best_estimator.coef_, [0], atol=1e-3)).squeeze()

        self.selected_features = np.array(train_samples.columns)[where_mask]

        self.support_ = np.where(where_mask, True, False)

        return self

    def _get_support_mask(self):
        return self.support_

    def transform(self, X):
        return X.loc[:, self._get_support_mask()]


class GeneticFeatureSelector(SelectorMixin):
    def __init__(self, estimator, scoring="f1", cv=None, max_features=20):
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
    def __init__(self, estimator, scoring="f1", cv=None, forward=True, floating=False, k_features="best"):
        super(SequentialFeatureSelector, self).__init__()
        self.estimator = estimator
        self.best_estimator = None
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


class PassthroughFeatureSelector(SelectorMixin):

    def __init__(self):
        super(PassthroughFeatureSelector, self).__init__()

    def fit(self, train_samples: pd.DataFrame, y=None):
        self.expected_shape = train_samples.shape
        self.selected_features = train_samples.columns
        self.support_ = np.ones((self.expected_shape[1]), dtype=bool)

    def _get_support_mask(self):
        return self.support_

    def transform(self, X):
        return X.loc[:, self._get_support_mask()]
