import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin


class DataframeTransformerWrapper(TransformerMixin):

    def __init__(self, transformer):
        self.transformer = transformer

    def fit(self, X: pd.DataFrame, y=None):
        self.transformer.fit(X,y)
        return self


    def transform(self, X: pd.DataFrame, y=None):

        X = X.copy()
        X[:] = self.transformer.transform(X)
        return X


class CorrelationFilter(TransformerMixin):

    def __init__(self, correlation=0.95):
        self.correlation = correlation


    def fit(self, X: pd.DataFrame, y=None):
        print("Calculating correlations")
        cor_matrix = X.corr().abs()
        print("Selecting features")
        upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
        self.to_select = X.columns[~upper_tri.apply(lambda x: x.max() > 0.95, axis=0)]
        return self

    def transform(self, X: pd.DataFrame, y=None):

        return X[self.to_select]


class VarianceFilter(TransformerMixin):

    def __init__(self, min_variance = 1e-10):

        self.min_variance = min_variance

    def fit(self, X: pd.DataFrame, y=None):
        print("Calculating variance")
        var = X.var()
        print("Selecting features")
        self.to_select = X.columns[var > self.min_variance]
        return self

    def transform(self, X: pd.DataFrame, y=None):

        return X[self.to_select]
