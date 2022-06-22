import os

from enum import Enum
from dataclasses import dataclass

import numpy as np
import cv2
import pandas as pd

class XRayLabel(Enum):

    Normal  = 'Normal'
    Lung_Opacity = 'Lung_Opacity'
    COVID = 'COVID'
    Viral_Pneumonia = 'Viral Pneumonia'
    SICK = "SICK"


class SubDataSet(Enum):

    ALL = "ALL"
    COVID_VS_HEALTHY = "COVID_VS_HEALTHY"
    SICK_VS_HEALTHY = "SICK_VS_HEALTHY"
    SICK = "SICK"


def get_subdataset_metadata(metadata: pd.DataFrame, subdataset: SubDataSet):

    if subdataset == SubDataSet.ALL:
        return metadata
    if subdataset == SubDataSet.COVID_VS_HEALTHY:
        return metadata[metadata.label.isin([XRayLabel.COVID.value, XRayLabel.Normal.value])]
    if subdataset == SubDataSet.SICK_VS_HEALTHY:
        metadata = metadata.copy()
        metadata.label = np.where(metadata.label == XRayLabel.Normal.value, XRayLabel.Normal.value, XRayLabel.SICK.value)
        return metadata
    if subdataset == SubDataSet.SICK:
        return metadata[metadata.label != XRayLabel.Normal.value]

    raise ValueError("Incorrect subdataset label")


def load_feature_sets(features_path, metadata_path):
    features = pd.read_csv(features_path, index_col=0)
    metadata = pd.read_csv(metadata_path, index_col=0)

    index_train, index_test = metadata[metadata.fold > 0].index, metadata[metadata.fold < 1].index

    features_train = features.loc[index_train, :]
    features_test = features.loc[index_test, :]

    return features, features_train, features_test, metadata

@dataclass
class XRayStudy:

    scan: np.ndarray = None
    mask: np.ndarray = None
    label: XRayLabel = None

class DatasetReader:

    def __init__(self, dataset_directory):
        self.dataset_directory = dataset_directory

    def load_file(self, name, mask=False):
        subdir = 'scans' if not mask else 'masks'
        filename = os.path.join(self.dataset_directory, subdir, name)
        return cv2.imread(filename, cv2.IMREAD_GRAYSCALE)[...,None]

    def load_study(self, name):
        scan = self.load_file(name, False)
        mask = self.load_file(name, True)
        return scan, mask

    def load_cases(self, metadata: pd.DataFrame):
        for row in metadata.iterrows():
            index, series = row
            scan, mask = self.load_study(series['id'])
            yield XRayStudy(scan, mask, XRayLabel(series['label']))