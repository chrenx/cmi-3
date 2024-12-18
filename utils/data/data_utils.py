import numpy as np


def quantile_clip_row(values, min_q=0.0, max_q=1.0):
    """Row-based operation performs quantile-based clipping on each row of a given matrix.
    If min_q=0 & max_q=1, that means there is no clipping for the row

    Args:
        values: dataframe (specimen, versioned_ensembl_gene_id)
        min_q (float): min quantil
        max_q (float): max quantile

    Returns:
        dataframe: clipped dataframe
    """
    ret = np.zeros_like(values)
    for i in range(values.shape[0]):
        row = values[i, :].copy()
        q_values = np.quantile(row, [min_q, max_q])
        row[row < q_values[0]] = q_values[0]
        row[row > q_values[1]] = q_values[1]
        ret[i, :] = row
    return ret

def median_normalize(values, ignore_zero=True, log=False):
    """Row-base operation.
    """

    tmp_values = values.copy()
    if ignore_zero:
        tmp_values[tmp_values == 0] = np.nan

    # Calculate the median with handling for all-NaN slices
    with np.errstate(invalid='ignore'):  # Suppress warnings for all-NaN slices
        print("ignore nan")
        nonzero_median = np.nanquantile(tmp_values, q=0.5, axis=1).astype(values.dtype)

    # Set median to 1 for rows where the median is NaN (to avoid division by NaN or zero)
    nonzero_median = np.where(np.isnan(nonzero_median), 1, nonzero_median)

    if log:
        ret = values - nonzero_median[:, None]
    else:
        ret = values / nonzero_median[:, None]
        
    ret = np.nan_to_num(ret, nan=0)
    return ret
