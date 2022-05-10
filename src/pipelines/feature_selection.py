from typing import NamedTuple, Union
import os

import neptune.new as neptune
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder
from gems.io import Json, Pickle

from src.feature_selection import SelectorType, get_selector_by_type
from src.pipelines.load_args import load_args
from src.data import load_feature_sets
from src.pipelines.pipeline_helpers import run_transformer_pipeline_on_train_and_test
from tracking.tracking_utils import log_metadata
from src.classification import get_stratified_folds

class FeatureSelectionPipelineConfig(NamedTuple):
    name: str
    params: dict

    @classmethod
    def config_from_json(cls, json: Union[str, dict]):
        if isinstance(json, str):
            json = Json.load(json)
        name = json['name']
        params = json['params']
        return cls(name, params)


class FeatureSelectionPipeline(TransformerMixin):

    def __init__(self, config: FeatureSelectionPipelineConfig, cv=None):
        super(FeatureSelectionPipeline, self).__init__()
        self.config = config
        self.selector_params = config.params
        self.selector_object = get_selector_by_type(SelectorType(config.name))(**self.selector_params, cv=cv)

    def fit(self, X, y=None):
        self.selector_object.fit(X, y)
        return self

    def transform(self, X, y=None):
        return self.selector_object.transform(X)

    def __repr__(self):
        return f"{self.config.name} selector with the following params: {self.selector_params}"


def init_neptune():
    from tracking.configs.feature_selection import project, api_token
    run = neptune.init(project=project, api_token=api_token)
    return run


if __name__ == "__main__":
    args = load_args()

    config = FeatureSelectionPipelineConfig.config_from_json(args.config)
    features, features_train, features_test, metadata = load_feature_sets(args.features, args.metadata)

    le = LabelEncoder()
    y_train = le.fit_transform(metadata.loc[features_train.index].label)
    cv = get_stratified_folds(features_train, y_train)

    pipeline = FeatureSelectionPipeline(config, cv)

    features_transformed, train_transformed, test_transformed = run_transformer_pipeline_on_train_and_test(pipeline,
                                                                                                           features,
                                                                                                           features_train,
                                                                                                           features_test,
                                                                                                           y_train)

    run = init_neptune()
    run_label = run._label

    output_path_dir = f'./experiments/Feature_Selection/{run_label}'
    os.makedirs(output_path_dir, exist_ok=True)

    output_path_features = os.path.join(output_path_dir, 'preprocessed_features.csv')
    output_path_pipeline = os.path.join(output_path_dir, 'pipeline.pkl')
    output_path_label_encoder = os.path.join(output_path_dir, 'label_encoder.pkl')

    features_transformed.to_csv(output_path_features)
    Pickle.save(output_path_pipeline, pipeline)
    Pickle.save(output_path_label_encoder, le)

    log_metadata(run, config, args, features, features_train, features_test, args.suffix)

    run['output_shape'] = features_transformed.shape
    run['artifacts/preprocessed_features'].upload(output_path_features)
    run['artifacts/pipeline'].upload(output_path_pipeline)
    run['artifacts/label_encoder'].upload(output_path_label_encoder)
