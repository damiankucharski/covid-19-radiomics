import os
from typing import NamedTuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin
from sklearn.pipeline import make_pipeline

import neptune.new as neptune

from gems.io import Json, Pickle

from src.preprocessing import CorrelationFilter, VarianceFilter
from src.preprocessing import DataframeTransformerWrapper as DTW
from src.pipelines.load_args import load_args


class PreprocessingPipelineConfig(NamedTuple):
    correlation: float = 0.95
    variance: float = 1e-10

    @classmethod
    def config_from_json(cls, json: Union[str, dict]):
        if isinstance(json, str):
            json = Json.load(json)

        return cls(**json)


class PreprocessingPipeline(TransformerMixin):

    def __init__(self, config: PreprocessingPipelineConfig):
        super(PreprocessingPipeline, self).__init__()
        self.config = config
        self.scaler = DTW(StandardScaler())
        self.variance_filter = VarianceFilter(min_variance=config.variance)
        self.correlation_filter = CorrelationFilter(correlation=config.correlation)

        self.pipeline = make_pipeline(self.scaler, self.variance_filter, self.correlation_filter)

    def fit(self, X, y=None):
        self.pipeline.fit(X, y)
        return self

    def transform(self, X, y=None):
        return self.pipeline.transform(X)

    def __repr__(self):
        return f"Preprocessing pipeline with the following config: {self.config.__repr__()}"


def init_neptune():
    from tracking.preprocessing import project, api_token
    run = neptune.init(project=project, api_token=api_token)
    return run


if __name__ == "__main__":
    run = init_neptune()
    args = load_args()

    run_label = run._label

    config = PreprocessingPipelineConfig.config_from_json(args.config)
    pipeline = PreprocessingPipeline(config)
    features = pd.read_csv(args.features, index_col=0)
    metadata = pd.read_csv(args.metadata, index_col=0)

    run['config'] = config._asdict()
    run['meta/features_source'] = args.features
    run['meta/metadata_source'] = args.metadata
    run['meta/features_shape'] = features.shape

    features_copy = features.copy()

    index_train, index_test = metadata[metadata.fold != 0].index, metadata[metadata.fold == 0].index

    features_train = features_copy.loc[index_train, :]
    features_test = features_copy.loc[index_test, :]

    run['meta/features_train_shape'] = features_train.shape
    run['meta/features_test_shape'] = features_test.shape

    train_transformed = pipeline.fit_transform(features_train)
    test_transformed = pipeline.transform(features_test)

    features_copy = features_copy.loc[:, train_transformed.columns]
    features_copy.loc[train_transformed.index, :] = train_transformed
    features_copy.loc[test_transformed.index, :] = test_transformed

    output_path_dir = f'./experiments/Preprocessing/{run_label}'
    os.makedirs(output_path_dir, exist_ok=True)

    output_path_features = os.path.join(output_path_dir, 'preprocessed_features.csv')
    output_path_pipeline = os.path.join(output_path_dir, 'pipeline.pkl')

    features_copy.to_csv()
    Pickle.save(output_path_pipeline, pipeline)

    run['output_shape'] = features_copy.shape
    run['artifacts/preprocessed_features'].upload(output_path_features)
    run['artifacts/pipeline'].upload(output_path_pipeline)
