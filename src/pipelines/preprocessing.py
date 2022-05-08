from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin

from typing import NamedTuple

from sklearn.pipeline import make_pipeline
from src.preprocessing import DataframeTransformerWrapper as DTW
from src.preprocessing import CorrelationFilter, VarianceFilter

class PreprocessingPipelineConfig(NamedTuple):

    correlation: float = 0.95
    variance: float = 1e-10


class PreprocessingPipeline(TransformerMixin):

    def __init__(self, config:PreprocessingPipelineConfig):

        self.scaler = DTW(StandardScaler())
        self.variance_filter = VarianceFilter(min_variance=config.variance)
        self.correlation_filter = CorrelationFilter(correlation=config.correlation)

        self.pipeline = make_pipeline(self.scaler, self.variance_filter, self.correlation_filter)

    def fit(self, X, y=None):
        self.pipeline.fit(X, y)
        return self

    def transform(self, X, y=None):
        return self.pipeline.transform(X)
