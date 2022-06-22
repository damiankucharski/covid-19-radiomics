import pandas as pd

def load_metadata(path = './src/metadata/metadata.csv'):
    return pd.read_csv(path, index_col=0)