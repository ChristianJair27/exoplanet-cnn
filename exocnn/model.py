"""Networks.

``AstroNet``: the best configuration of Shallue & Vanderburg (2018, Fig. 7): a global column of five blocks
(two conv5 layers with 16/32/64/128/256 maps, then maxpool5 stride 2), a local column of two blocks (two
conv5 layers with 16/32 maps, then maxpool7 stride 2), four FC-512 layers and a sigmoid output.

``ExtendedNet``: AstroNet's two columns plus diagnostic branches in the spirit of ExoMiner (Valizadegan et
al. 2022, Sec. 3.3): one local-type column applied with shared weights to the odd and the even view, whose
features are *subtracted*; a local-type column for the secondary-eclipse view; a small MLP for scalar DV
features. As in ExoMiner, each branch ends in its own FC layer so that every branch hands the same number
of features to the shared FC block.
"""
from __future__ import annotations

import torch
from torch import nn


def column(n_in: int, blocks, pool: int) -> nn.Sequential:
    layers, c = [], 1
    for maps in blocks:
        layers += [nn.Conv1d(c, maps, 5, padding=2), nn.ReLU(), nn.Conv1d(maps, maps, 5, padding=2), nn.ReLU(),
                   nn.MaxPool1d(pool, stride=2)]
        c = maps
    layers.append(nn.Flatten())
    return nn.Sequential(*layers)


def flat_size(col: nn.Module, n: int) -> int:
    with torch.no_grad():
        return col(torch.zeros(1, 1, n)).shape[1]


GLOBAL_BLOCKS = (16, 32, 64, 128, 256)
LOCAL_BLOCKS = (16, 32)


class AstroNet(nn.Module):
    def __init__(self, n_global=2001, n_local=201, use_global=True, use_local=True, fc=512, n_fc=4,
                 dropout=0.0):
        super().__init__()
        self.use_global, self.use_local = use_global, use_local
        width = 0
        if use_global:
            self.g = column(n_global, GLOBAL_BLOCKS, 5)
            width += flat_size(self.g, n_global)
        if use_local:
            self.l = column(n_local, LOCAL_BLOCKS, 7)
            width += flat_size(self.l, n_local)
        head = []
        for _ in range(n_fc):
            head += [nn.Linear(width, fc), nn.ReLU(), nn.Dropout(dropout)]
            width = fc
        head.append(nn.Linear(width, 1))
        self.head = nn.Sequential(*head)

    def forward(self, b: dict) -> torch.Tensor:
        feats = []
        if self.use_global:
            feats.append(self.g(b["global"][:, None]))
        if self.use_local:
            feats.append(self.l(b["local"][:, None]))
        return self.head(torch.cat(feats, 1)).squeeze(1)   # logits


class ExtendedNet(nn.Module):
    def __init__(self, n_global=2001, n_local=201, n_scalars=16, branch=128, fc=512, n_fc=3, dropout=0.3,
                 use_scalars=True):
        super().__init__()
        self.use_scalars = use_scalars

        def proj(col, n):
            return nn.Sequential(nn.Linear(flat_size(col, n), branch), nn.ReLU())

        self.g = column(n_global, GLOBAL_BLOCKS, 5)
        self.g_fc = proj(self.g, n_global)
        self.l = column(n_local, LOCAL_BLOCKS, 7)
        self.l_fc = proj(self.l, n_local)
        self.oe = column(n_local, LOCAL_BLOCKS, 7)          # shared by odd and even
        self.oe_fc = proj(self.oe, n_local)
        self.s = column(n_local, LOCAL_BLOCKS, 7)
        self.s_fc = proj(self.s, n_local)
        width = 4 * branch
        if use_scalars:
            self.sc = nn.Sequential(nn.Linear(n_scalars, 64), nn.ReLU(), nn.Linear(64, branch), nn.ReLU())
            width += branch
        head = []
        for _ in range(n_fc):
            head += [nn.Linear(width, fc), nn.ReLU(), nn.Dropout(dropout)]
            width = fc
        head.append(nn.Linear(width, 1))
        self.head = nn.Sequential(*head)

    def forward(self, b: dict) -> torch.Tensor:
        odd = self.oe(b["odd"][:, None])
        even = self.oe(b["even"][:, None])
        feats = [
            self.g_fc(self.g(b["global"][:, None])),
            self.l_fc(self.l(b["local"][:, None])),
            self.oe_fc(odd - even),                         # non-trainable subtraction (ExoMiner)
            self.s_fc(self.s(b["secondary"][:, None])),
        ]
        if self.use_scalars:
            feats.append(self.sc(b["scalars"]))
        return self.head(torch.cat(feats, 1)).squeeze(1)


def build(name: str, n_scalars: int = 16) -> nn.Module:
    if name == "astronet":
        return AstroNet()
    if name == "global":
        return AstroNet(use_local=False)
    if name == "local":
        return AstroNet(use_global=False)
    if name == "extended":
        return ExtendedNet(n_scalars=n_scalars)
    if name == "extended_noscalars":
        return ExtendedNet(n_scalars=n_scalars, use_scalars=False)
    raise ValueError(name)
