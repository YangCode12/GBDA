import numpy as np
import torch
import os
from torch import nn
import torch.nn.functional as F
from metrics import *
from utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class SE_Block(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, T = x.size()
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1)
        return x * y.expand_as(x)

class autocoder(nn.Module):
    def __init__(self, in_channels=62, latent_dim=128):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 128, kernel_size=1)
        self.se1 = SE_Block(128, reduction=8)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(128 * 5, latent_dim)

    def forward(self, x):
        h = self.relu(self.conv1(x))
        h = h + self.se1(h)
        h = self.relu(self.conv2(h))
        h = h.view(x.size(0), -1)
        return self.fc(h)

class model(nn.Module):
    def __init__(self, classs_num=3):
        super(model, self).__init__()
        self.autocoder = autocoder(62, 128)
        self.mmd_loss = MMDLoss(kernel_type='rbf',
                                kernel_mul=2.0,
                                kernel_num=5,
                                fix_sigma=None)
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, classs_num)
        )

    def forward(self, data_source, data_target=None, label_source=None, gb_weights=None):
        if self.training:
            feat_s = self.autocoder(data_source)
            feat_t = self.autocoder(data_target)

            mmdloss = self.mmd_loss(feat_s, feat_t)

            logits = self.classifier(feat_s)
            if gb_weights is not None:
                nll = F.cross_entropy(logits, label_source, reduction='none')
                classloss = (gb_weights * nll).sum() / gb_weights.sum()
            else:
                classloss = F.cross_entropy(logits, label_source)

            return mmdloss, classloss, logits

        else:
            feat = self.autocoder(data_source)
            logits = self.classifier(feat)
            return logits
