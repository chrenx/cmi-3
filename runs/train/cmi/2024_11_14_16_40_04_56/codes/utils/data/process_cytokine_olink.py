import argparse, os

import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

from utils.data.data_utils import median_normalize, quantile_clip_row

def process_cytokine_olink(filename, dpath=None, df=None):
    """
    df: (specimen, protein_id) includes both train, target, and challenge
    """
    n_components = 32
    seed = 42

    if df is None:
        df = pd.read_parquet(dpath)
    print("dataframe shape:", df.shape)

    orig_idx = df.index
    data = df.to_numpy()
    data = np.nan_to_num(data, nan=0)

    data = quantile_clip_row(data, min_q=0.0, max_q=1)

    # thres = 5
    # data[data < thres] = np.expm1(data[data < thres])

    data = np.log1p(median_normalize(data)) 
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    x_reduced = svd.fit_transform(data)
    scalar = StandardScaler()
    scalar.fit(x_reduced)
    x_reduced = scalar.transform(x_reduced)
    new_df = pd.DataFrame(x_reduced, index=orig_idx)
    
    save_path = os.path.join("data/processed_data", filename)
    new_df.to_parquet(save_path)
    print("processed data is saved to", save_path)
    return x_reduced

