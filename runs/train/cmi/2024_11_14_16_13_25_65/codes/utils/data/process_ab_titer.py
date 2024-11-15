import argparse, os

import pandas as pd
import numpy as np

from utils.data.data_utils import median_normalize


def process_ab_titer(filename, dpath=None, df=None):
    """
    df: (specimen, versioned_ensembl_gene_id) includes both train, target, and challenge
    """
    seed = 42

    if df is None:
        df = pd.read_parquet(dpath)
    print("dataframe shape:", df.shape)

    orig_idx = df.index
    data = df.to_numpy()
    data = np.nan_to_num(data, nan=0)

    data = np.log1p(median_normalize(data)) 

    new_df = pd.DataFrame(data, index=orig_idx)
    
    save_path = os.path.join("data/processed_data", filename)
    new_df.to_parquet(save_path)
    print("processed data is saved to", save_path)
    return new_df
