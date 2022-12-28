'''
mainly from github chenxin-dlut TransT
Now this file is used as a high-performance MLP...
'''
import math
import torch
from torch import nn, Tensor
import copy
from typing import Optional
import torch.nn.functional as F
import numpy as np
from pointnet_utils import knn_point, group_operation
import sys
sys.path.append('../..')

class TransT(nn.Module):
    def __init__(self, d_model=384, concat=False):
        super().__init__()
        self.s11 = attn_module(d_model=d_model,no_linear=True, concat=concat)
        self.s12 = attn_module(d_model=d_model,no_linear=True, concat=concat)
        self.c11 = attn_module(d_model=d_model,concat=concat)
        self.c12 = attn_module(d_model=d_model,concat=concat)

    def forward(self,src1, pos1, src2, pos2, attn):
        src11 = self.s11(src1, pos1, src1, pos1,attn)
        src12 = self.s12(src2, pos2, src2, pos2,attn)
        result1 = self.c11(src11, pos1, src12, pos2,attn)
        result2 = self.c12(src12, pos2, src11, pos1,attn)
        return result1, result2

class attn_module(nn.Module):
    def __init__(self, d_model=384, no_linear=False, only_pos=False, qk_mask=None, nhead=8, dim_feedforward=1024, dropout=0.1, activation="relu", concat=False):
        super().__init__()
        if concat:
            self.attn = nn.MultiheadAttention(72, nhead, vdim=d_model, dropout=dropout)
            self.newlq = nn.Linear(d_model, 72)
            self.newlk = nn.Linear(d_model, 72)
            self.outlv = nn.Linear(72, d_model)
        else:
            self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.no_linear = no_linear
        self.only_pos = only_pos
        self.qk_mask = qk_mask
        self.concat = concat
        if not no_linear:
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.linear2 = nn.Linear(dim_feedforward, d_model)
            self.dropout2 = nn.Dropout(dropout)
            self.dropout3 = nn.Dropout(dropout)
            self.norm2 = nn.LayerNorm(d_model)
            self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, src1_ori, pos1_ori, src2_ori, pos2_ori, attn=True):
        '''
        :param src1: q [B, C, N]
        :param pos1: positional embedding[B, C, N]
        :param src2: k, v [B, C, M]
        :param pos2: positional embedding[B, C, M]
        :param qk_mask: hand kp neighbors mask
        :return: same shape as q
        '''
        src1 = src1_ori.permute(2, 0, 1)
        pos1 = pos1_ori.permute(2, 0, 1)
        src2 = src2_ori.permute(2, 0, 1)
        pos2 = pos2_ori.permute(2, 0, 1)
        if self.concat:
            src12, weight = self.attn(self.with_pos_embed(self.newlq(src1), pos1), self.with_pos_embed(self.newlk(src2), pos2),
                                      value=src2, attn_mask=self.qk_mask)
            src1_new = src1 + self.outlv(self.dropout1(src12))

        else:
            src12, weight = self.attn(self.with_pos_embed(src1, pos1), self.with_pos_embed(src2, pos2),
                             value=src2, attn_mask=self.qk_mask)
            src1_new = src1 + self.dropout1(src12)
        if not attn:
            src1_new = src1
        src1_new = self.norm1(src1_new)
        if not self.no_linear:
            src13 = self.linear2(self.dropout2(self.activation(self.linear1(src1_new))))
            src1_new = src1_new + self.dropout3(src13)
            src1_new = self.norm2(src1_new)
        return src1_new.permute(1,2,0)

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, normalize=True):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.normalize = normalize
        if normalize is False:
            raise ValueError("normalize should be True if scale is passed")

    def forward(self, coor):
        # ref to https://arxiv.org/pdf/2003.08934v2.pdf

        # normal the coor into [-1, 1], batch wise
        if self.normalize:
            normal_coor = 2 * ((coor - coor.min()) / (coor.max() - coor.min())) - 1

        # define sin wave freq
        freqs = torch.arange(self.num_pos_feats, dtype=torch.float).cuda()
        freqs = np.pi * (2**freqs)

        freqs = freqs.view(*[1]*len(normal_coor.shape), -1) # 1 x 1 x 1 x D
        normal_coor = normal_coor.unsqueeze(-1) # B x 3 x N x 1
        k = normal_coor * freqs # B x 3 x N x D
        s = torch.sin(k) # B x 3 x N x D
        c = torch.cos(k) # B x 3 x N x D
        x = torch.cat([s,c], -1) # B x 3 x N x 2D
        pos = x.transpose(-1,-2).reshape(coor.shape[0], -1, coor.shape[-1]) # B 6D N
        # zero_pad = torch.zeros(x.size(0), 2, x.size(-1)).cuda()
        # pos = torch.cat([x, zero_pad], dim = 1)
        # pos = self.pos_embed_wave(x)
        return pos

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
