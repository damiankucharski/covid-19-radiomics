from src.data import DatasetReader
from src.feature_extraction import PyradiomicsFeatureExtractor

import os
import pandas as pd

from tqdm import tqdm


if __name__ == "__main__":

    metadata = pd.read_csv('./src/metadata/metadata.csv')

    import logging
    # set level for all classes
    logger = logging.getLogger("radiomics")
    logger.setLevel(logging.ERROR)

    reader = DatasetReader('./Data/Dataset')
    print(os.path.exists('src/metadata/params.yaml'))
    extractor = PyradiomicsFeatureExtractor('src/metadata/params.yaml')

    features = []
    for case in tqdm(reader.load_cases(metadata)):
        features.append(extractor.calculate_features(case))


    features = pd.concat(features, axis=1).T
    features.index = metadata['id']

    features.to_csv('./src/extracted_features/features.csv')
