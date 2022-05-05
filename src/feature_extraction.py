import six

import SimpleITK as sitk
import numpy as np
import pandas as pd
from radiomics import featureextractor

from src.data import XRayStudy

class PyradiomicsFeatureExtractor:

    def __init__(self, config_path):
        self.config_path = config_path
        self.extractor = featureextractor.RadiomicsFeatureExtractor(config_path)

    def calculate_features(self, study: XRayStudy):
        result = self.extractor.execute(sitk.GetImageFromArray(study.scan), sitk.GetImageFromArray(study.mask), voxelBased=False)
        self.extractor.enableAllImageTypes()
        names = []
        vals = []
        for key, val in six.iteritems(result):
            if 'diagn' not in key:
                names.append(key)
                vals.append(val)
        return pd.Series(data=vals, index = names)

    def calculate_feature_maps(self, study: XRayStudy):
        mask_array = study.mask
        result = self.extractor.execute(sitk.GetImageFromArray(study.scan), sitk.GetImageFromArray(study.mask), voxelBased=True)
        res_dict = {}
        where_mask = np.where(mask_array > 0)
        xa_m, xb_m, ya_m, yb_m = where_mask[0].min(), where_mask[0].max(), where_mask[1].min(), where_mask[1].max()
        for key, val in six.iteritems(result):
            if isinstance(val, sitk.Image):  # Feature map
                fm_array = np.zeros_like(mask_array)
                fm = sitk.GetArrayFromImage(val)
                fm_array[xa_m:xa_m+fm.shape[0], ya_m:ya_m+fm.shape[1]] = fm
                res_dict[key] = fm_array
        return res_dict