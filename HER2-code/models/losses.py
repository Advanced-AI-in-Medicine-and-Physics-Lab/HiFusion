import numpy as np
import torch


# Loss function: MSE + PCC
def pcc_loss(output, target):
    # Convert target to float type
    target = target.float()
    x = output - output.mean(dim=1, keepdim=True)
    y = target - target.mean(dim=1, keepdim=True)

    covariance = (x * y).sum(dim=1)
    bessel_corrected_variance_x = (x ** 2).sum(dim=1)
    bessel_corrected_variance_y = (y ** 2).sum(dim=1)

    pcc = covariance / torch.sqrt(bessel_corrected_variance_x * bessel_corrected_variance_y + 1e-8)
    return 1 - pcc.mean()  # 1 - PCC as the loss



def calculate_pcc(pred, target):
    target = target.float()
    x = pred - pred.mean(dim=1, keepdim=True)
    y = target - target.mean(dim=1, keepdim=True)

    covariance = (x * y).sum(dim=1)
    bessel_corrected_variance_x = (x ** 2).sum(dim=1)
    bessel_corrected_variance_y = (y ** 2).sum(dim=1)

    pcc = covariance / torch.sqrt(bessel_corrected_variance_x * bessel_corrected_variance_y + 1e-8)
    return pcc.mean()


