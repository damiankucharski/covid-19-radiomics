import pandas as pd

def load_features(path = './src/extracted_features/features.csv'):
    return pd.read_csv(path, index_col=0)