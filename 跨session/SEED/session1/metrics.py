import torch
import torch.nn.functional as F
from torch import Tensor
from scipy.signal import welch
from sklearn.metrics import accuracy_score
from typing import Tuple
import numpy as np

def _welch_psd(x, fs=200):
    device = x.device
    x = x.detach().cpu().numpy()
    B, C, T = x.shape
    win = min(256, T)                 # 窗长不能大于信号长度
    nfft = win                        # 也可设 2 的幂，但 win 已足够
    _, Pxx = welch(x[0, 0], fs=fs, nperseg=win, nfft=nfft)  # 先取一条样本看长度
    F = Pxx.shape[0]                  # 实际频点个数
    psd = np.zeros((B, C, F))
    for b in range(B):
        for c in range(C):
            _, Pxx = welch(x[b, c], fs=fs, nperseg=win, nfft=nfft)
            psd[b, c] = Pxx
    return torch.from_numpy(psd).float().to(device)  # ③ 回到原设备

def mse(x_hat: Tensor, x: Tensor) :
    """平均平方误差"""
    return F.mse_loss(x_hat, x)

def rmse(x_hat: Tensor, x: Tensor): 
    return torch.sqrt(mse(x_hat, x))

def cosine_sim(x_hat: Tensor, x: Tensor, dim: int = -1):
    """trial 内平均余弦相似度"""
    return F.cosine_similarity(x_hat, x, dim=dim).mean()


def psd_mse(x_hat: Tensor, x: Tensor, fs: int = 200):
    """log-PSD 的 MSE"""
    psd_hat = _welch_psd(x_hat, fs)
    psd_true = _welch_psd(x, fs)
    return F.mse_loss(psd_hat.log(), psd_true.log())

def psnr(x, x_rec, max_val=3.0):
    """
    x, x_rec: (B, C, T)  tensor
    return:  batch 平均 PSNR (dB)
    """
    mse = F.mse_loss(x_rec, x, reduction='mean')        # 已平均
    if mse == 0:
        return torch.tensor(float('inf'), device=x.device)
    return 10 * torch.log10(max_val**2 / mse)



