import pandas as pd

def load_metadata():
    return pd.read_csv('./src/metadata/metadata.csv', index_col=0)