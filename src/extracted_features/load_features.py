import pandas as pd

def load_features():
    return pd.read_csv('./src/extracted_features/features.csv', index_col=0)