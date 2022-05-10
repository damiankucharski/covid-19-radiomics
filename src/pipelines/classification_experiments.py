from typing import NamedTuple, Union


from src.pipelines.names import CLASSIFICATION_PIPELINE_CONFIG, PREPROCESSING_PIPELINE_CONFIG,\
    FEATURE_SELECTION_PIPELINE_CONFIG
from src.pipelines.feature_selection import FeatureSelectionPipeline
from src.pipelines.preprocessing import PreprocessingPipeline
from src.pipelines.classification import ClassificationPipeline


from src.feature_selection import PassthroughFeatureSelector

import neptune.new as neptune
from gems.io import Json


from argparse import ArgumentParser

class ClassificationExperimentConfig(NamedTuple):

    ClassificationPipelineConfig: CLASSIFICATION_PIPELINE_CONFIG
    FeatureSelectionPipelineConfig: FEATURE_SELECTION_PIPELINE_CONFIG
    PreprocessingPipelineConfig: PREPROCESSING_PIPELINE_CONFIG = None


    @classmethod
    def config_from_json(cls, json: Union[str, dict]):
        if isinstance(json, str):
            json = Json.load(json)

        preprocessing_config = json.get(PREPROCESSING_PIPELINE_CONFIG, None)
        feature_selection_pipeline_config = json[FEATURE_SELECTION_PIPELINE_CONFIG]
        classification_pipeline_config = json[CLASSIFICATION_PIPELINE_CONFIG]

        return cls(classification_pipeline_config, feature_selection_pipeline_config, preprocessing_config)

class ClassificationExperiment:

    def __init__(self, config: ClassificationExperimentConfig):
        self.config = config


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('-c', '--config', default='./configs/classification_experiments/example_config.json')
    args = parser.parse_args()

