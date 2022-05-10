import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from typing import Union

def create_dataset_metadata(base_metadata: pd.DataFrame, train_size: Union[int, float]):
    base_metadata['fold'] = np.zeros((len(base_metadata)), dtype=int)
    y = LabelEncoder().fit_transform(base_metadata.label)

    if isinstance(train_size, float):
        test_size = 1 - train_size
    else:
        test_size = len(y) - train_size

    metadata_train, metadata_test, y_train, y_test = train_test_split(base_metadata, y,
                                                                      test_size=test_size, shuffle=True, stratify=y,
                                                                      random_state=1992)
    skf = StratifiedKFold()
    for fold_no, (index_train, index_test) in enumerate(skf.split(metadata_train, y_train), start=1):
        base_metadata.loc[metadata_train.index[index_test], 'fold'] = fold_no

    return base_metadata