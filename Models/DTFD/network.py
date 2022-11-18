import os
import torch
import torch.nn as nn

class residual_block(nn.Module):
    def __init__(self, nChn=512):
        super(residual_block, self).__init__()
        self.block = nn.Sequential(
                nn.Linear(nChn, nChn, bias=False),
                nn.ReLU(inplace=False),
                nn.Linear(nChn, nChn, bias=False),
                nn.ReLU(inplace=False),
            )
    def forward(self, x):
        tt = self.block(x)
        x = x + tt
        return x

class DimReduction(nn.Module):
    def __init__(self, n_channels, m_dim=512, numLayer_Res=0):
        super(DimReduction, self).__init__()
        if n_channels == m_dim:
            self.fc1 = nn.Identity()
            self.relu1 = nn.Identity()
        else:
            self.fc1 = nn.Linear(n_channels, m_dim, bias=False)
            self.relu1 = nn.ReLU(inplace=False)
            self.numRes = numLayer_Res

        self.resBlocks = []
        for ii in range(numLayer_Res):
            self.resBlocks.append(residual_block(m_dim))
        self.resBlocks = nn.Sequential(*self.resBlocks)

    def forward(self, x):

        x = self.fc1(x)
        x = self.relu1(x)

        if self.numRes > 0:
            x = self.resBlocks(x)

        return x


def get_cam_1d(classifier, features):
    # tweight = list(classifier.parameters())[-2]
    tweight = classifier.fc.weight
    cam_maps = torch.einsum('bgf,cf->bcg', [features, tweight])
    return cam_maps