import torch
import torch.nn as nn


def chaos_loss(pred, target, lam=1.0):
    """
    Custom loss function for chaotic time series.
    Combines MSE with a divergence penalty between neighboring predictions.
    """
    mse = nn.MSELoss()(pred, target)

    # Temporal divergence: penalize difference in prediction trajectory smoothness
    pred_diff = pred[:, 1:] - pred[:, :-1]
    target_diff = target[:, 1:] - target[:, :-1]
    divergence = nn.MSELoss()(pred_diff, target_diff)

    return mse + lam * divergence
