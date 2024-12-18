import torch

def correlation_score(y_pred, y_true):
    y_true_centered = y_true - torch.mean(y_true, dim=1)[:, None]
    y_pred_centered = y_pred - torch.mean(y_pred, dim=1)[:, None]
    cov_tp = torch.sum(y_true_centered * y_pred_centered, dim=1) / (y_true.shape[1] - 1)
    var_t = torch.sum(y_true_centered ** 2, dim=1) / (y_true.shape[1] - 1)
    var_p = torch.sum(y_pred_centered ** 2, dim=1) / (y_true.shape[1] - 1)
    score = cov_tp / torch.sqrt(var_t * var_p)
    return -torch.mean(score)


def row_normalize(v):
    mu = torch.mean(v, dim=1)
    sigma = torch.std(v, dim=1)
    return (v - mu[:, None]) / sigma[:, None]
