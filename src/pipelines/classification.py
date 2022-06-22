from typing import NamedTuple, Union

from sklearn.base import TransformerMixin
from gems.io import Json
import neptune.new as neptune

from src.classification import get_classifier_by_type, ClassifierType


class ClassificationPipelineConfig(NamedTuple):
    name: str
    params: dict

    @classmethod
    def config_from_json(cls, json: Union[str, dict]):
        if isinstance(json, str):
            json = Json.load(json)
        name = json['name']
        params = json['params']
        return cls(name, params)


class ClassificationPipeline(TransformerMixin):

    def __init__(self, config: ClassificationPipelineConfig):
        super(ClassificationPipeline, self).__init__()

        self.config = config
        self.classifier_params = config.params
        self.classifier = get_classifier_by_type(ClassifierType(config.name))(config.params)


    def fit(self, X, y):
        self.classifier.fit(X, y)
        return self

    def predict(self, X, y=None):
        return self.classifier.predict(X)

    def __repr__(self):
        return f"{self.config.name} classifier with the following params: {self.classifier_params}"