import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import trunc_normal_


class CNN(nn.Module):

    def __init__(self, embed_dim=64, nhead=8, num_encoder_layers=6, dim_feedforward=1024,
                 dropout=0.1, activation="relu", num_classes=23073):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.conv = ConvModule(drop_rate=dropout)

        self.proj = nn.Linear(141, 1)

        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * 4)),
            nn.BatchNorm1d(int(embed_dim * 4)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(int(embed_dim * 4), int(embed_dim * 4)),
            nn.BatchNorm1d(int(embed_dim * 4)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(int(embed_dim * 4), num_classes)
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, x):
        N = x.shape[0]
        if x.shape[1] == 2:
            x = x[:, 1:, :]

        x = x / 100
        x = self.conv(x)

        # flatten NxCxL to LxNxC
        # x = x.permute(2, 0, 1).contiguous()
        x = self.proj(x).flatten(1)

        logits = self.cls_head(x)
        return logits


class ConvModule(nn.Module):
    def __init__(self, drop_rate=0.):
        super().__init__()
        self.drop_rate = drop_rate

        self.conv1 = nn.Conv1d(1, 64, kernel_size=35, stride=2, padding=17)
        self.bn1 = nn.BatchNorm1d(64)
        self.act1 = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = Layer(64, 64, kernel_size=3, stride=2, downsample=True)
        self.layer2 = Layer(64, 128, kernel_size=3, stride=2, downsample=True)
        # self.layer3 = Layer(256, 256, kernel_size=3, stride=2, downsample=True)
        self.maxpool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        x = self.maxpool2(x)
        return x

class Layer(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, stride, downsample):
        super(Layer, self).__init__()
        self.block1 = BasicBlock(inchannel, outchannel, kernel_size=kernel_size, stride=stride, downsample=downsample)
        self.block2 = BasicBlock(outchannel, outchannel, kernel_size=kernel_size, stride=1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x

class BasicBlock(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, stride, downsample=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(outchannel)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(outchannel, outchannel, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(outchannel)
        self.act2 = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, kernel_size=1, stride=2),
            nn.BatchNorm1d(outchannel)
        ) if downsample else None

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act2(x)
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def with_pos_embed(tensor, pos):
    return tensor if pos is None else tensor + pos
