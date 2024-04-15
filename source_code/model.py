import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import trunc_normal_


class CPICANN_main(nn.Module):

    def __init__(self, embed_dim=64, nhead=8, num_encoder_layers=6, dim_feedforward=1024,
                 dropout=0.1, activation="relu", num_classes=273):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.conv = convEncoder(drop_rate=dropout)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, 142))

        encoder_layer = TransformerEncoderLayer(embed_dim, nhead, dim_feedforward, dropout, activation)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        self.norm_after = nn.LayerNorm(embed_dim)

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
        self.init_weights()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def init_weights(self):
        trunc_normal_(self.cls_token, std=.02)

        self.pos_embed.requires_grad = False

        pos_embed = get_1d_sincos_pos_embed_from_grid(self.embed_dim, np.array(range(self.pos_embed.shape[2])))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).T.unsqueeze(0))

    def bce_fineTune_init_weights(self):
        for p in self.conv.parameters():
            p.requires_grad = False

        for p in self.encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.cls_head.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def forward(self, x):
        N = x.shape[0]
        if x.shape[1] == 2:
            x = x[:, 1:, :]

        x = x / 100
        x = self.conv(x)

        # flatten NxCxL to LxNxC
        x = x.permute(2, 0, 1).contiguous()

        cls_token = self.cls_token.expand(-1, N, -1)
        x = torch.cat((cls_token, x), dim=0)

        pos_embed = self.pos_embed.permute(2, 0, 1).contiguous().repeat(1, N, 1)
        feats = self.encoder(x, pos_embed)
        feats = self.norm_after(feats)
        logits = self.cls_head(feats[0])
        return logits


class ConvBasicBlock(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, stride, downsample=False):
        super(ConvBasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(outchannel)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(outchannel, outchannel, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
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


class ConvLayer(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size, stride, downsample):
        super(ConvLayer, self).__init__()
        self.block1 = ConvBasicBlock(inchannel, outchannel, kernel_size=kernel_size, stride=stride, downsample=downsample)
        self.block2 = ConvBasicBlock(outchannel, outchannel, kernel_size=kernel_size, stride=1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x

class convEncoder(nn.Module):
    def __init__(self, drop_rate=0.):
        super().__init__()
        self.drop_rate = drop_rate

        self.conv1 = nn.Conv1d(1, 64, kernel_size=35, stride=2, padding=17)
        self.bn1 = nn.BatchNorm1d(64)
        self.act1 = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = ConvLayer(64, 64, kernel_size=3, stride=2, downsample=True)
        self.layer2 = ConvLayer(64, 128, kernel_size=3, stride=2, downsample=True)
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

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, pos):
        output = src

        for layer in self.layers:
            output = layer(output, pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, pos):
        q = k = with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def with_pos_embed(tensor, pos):
    return tensor if pos is None else tensor + pos


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out).astype(np.float32)  # (M, D/2)
    emb_cos = np.cos(out).astype(np.float32)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

