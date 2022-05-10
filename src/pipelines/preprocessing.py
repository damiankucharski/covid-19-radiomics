import os
from typing import NamedTuple, Union

from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin
from sklearn.pipeline import make_pipeline

import neptune.new as neptune

from gems.io import Json, Pickle

from src.preprocessing import CorrelationFilter, VarianceFilter
from src.preprocessing import DataframeTransformerWrapper as DTW
from src.pipelines.load_args import load_args
from src.pipelines.pipeline_helpers import run_transformer_pipeline_on_train_and_test
from src.data import load_feature_sets
from tracking.tracking_utils import log_metadata

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
    from tracking.configs.preprocessing import project, api_token
    run = neptune.init(project=project, api_token=api_token)
    return run


if __name__ == "__main__":

    args = load_args()

    config = PreprocessingPipelineConfig.config_from_json(args.config)
    pipeline = PreprocessingPipeline(config)


    features, features_train, features_test, metadata = load_feature_sets(args.features, args.metadata)

    features_transformed, train_transformed, test_transformed = run_transformer_pipeline_on_train_and_test(pipeline,
                                                                                                           features,
                                                                                                           features_train,
                                                                                                           features_test)

    run = init_neptune()
    run_label = run._label

    output_path_dir = f'./experiments/Preprocessing/{run_label}'
    os.makedirs(output_path_dir, exist_ok=True)

    output_path_features = os.path.join(output_path_dir, 'preprocessed_features.csv')
    output_path_pipeline = os.path.join(output_path_dir, 'pipeline.pkl')

    features_transformed.to_csv(output_path_features)
    Pickle.save(output_path_pipeline, pipeline)


    log_metadata(run, config, args, features, features_train, features_test, args.suffix)

    run['output_shape'] = features_transformed.shape
    run['artifacts/preprocessed_features'].upload(output_path_features)
    run['artifacts/pipeline'].upload(output_path_pipeline)
