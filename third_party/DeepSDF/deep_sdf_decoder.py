#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch.nn as nn
import torch
import torch.nn.functional as F
import os
import numpy as np

class Decoder(nn.Module):
    def __init__(
        self,
        latent_size,
        dims,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        xyz_in_all=None,
        use_tanh=False,
        latent_dropout=False,
    ):
        super(Decoder, self).__init__()

        def make_sequence():
            return []

        dims = [latent_size + 3] + dims + [1]

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        for l in range(0, self.num_layers - 1):
            if l + 1 in latent_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]
                if self.xyz_in_all and l != self.num_layers - 2:
                    out_dim -= 3

            if weight_norm and l in self.norm_layers:
                setattr(self, "lin" + str(l), nn.utils.weight_norm(nn.Linear(dims[l], out_dim)))
            else:
                setattr(self, "lin" + str(l), nn.Linear(dims[l], out_dim))

            if (not weight_norm) and self.norm_layers is not None and l in self.norm_layers:
                setattr(self, "bn" + str(l), nn.LayerNorm(out_dim))

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()
        self.training = False

    # input: N x (L+3)
    def forward(self, input, epoch=None):
        # input = torch.FloatTensor(np.loadtxt('/home/hewang/ym/Curriculum-DeepSDF/input.txt')).cuda()
        
        xyz = input[:, -3:]

        if input.shape[1] > 3 and self.latent_dropout:
            latent_vecs = input[:, :-3]
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
            x = torch.cat([latent_vecs, xyz], 1)
        else:
            x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if l in self.latent_in:
                x = torch.cat([x, input], 1)
            elif l != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], 1)
            x = lin(x)
            # last layer Tanh
            if l == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)
            if l < self.num_layers - 2:
                if self.norm_layers is not None and l in self.norm_layers and not self.weight_norm:
                    bn = getattr(self, "bn" + str(l))
                    x = bn(x)
                x = self.relu(x)
                if self.dropout is not None and l in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if hasattr(self, "th"):
            x = self.th(x)
        # np.savetxt('sdf.txt', x[:, 0].detach().cpu().numpy())
        # exit(0)

        return x

class SDF():
    def __init__(self, cfg):
        self.use_gt_latent_code = cfg['network']['use_gt_latent_code']
        latent_size = 256
        self.SDFDecoder = Decoder(latent_size, **cfg['network']["NetworkSpecs"])
        self.SDFDecoder = torch.nn.DataParallel(self.SDFDecoder)
        saved_model_state = torch.load(
            '/home/hewang/ym/Curriculum-DeepSDF/examples/bottle_sim/ModelParameters/2000.pth'
        )
        self.SDFDecoder.load_state_dict(saved_model_state["model_state_dict"])
        self.SDFDecoder = self.SDFDecoder.module.cuda()
        self.latent_code_dir = '/mnt/data/hewang/h2o_data/SDF/Reconstructions/bottle_sim/2000/Codes'
        self.normalization_dir = '/mnt/data/hewang/h2o_data/SDF/NormalizationParameters/bottle_sim'
        print('Load SDF!')
    
    def load_obj(self, instance):
        if self.use_gt_latent_code:
            latent_code_pth = os.path.join(self.latent_code_dir, instance[:5] + '.pth')
        else:
            latent_code_pth = os.path.join(self.latent_code_dir, instance + '.pth')
        self.latent_code = torch.load(latent_code_pth)[0][0].cuda() # 1, 1, L
        
        normalization_pth = os.path.join(self.normalization_dir, instance[:5] + '.npz')
        normalization_params = np.load(normalization_pth)
        self.normalization_scale = torch.FloatTensor(normalization_params['scale']).cuda()
        print('Load latent code of ', latent_code_pth)
    
    def get_penetrate_from_sdf(self, query, threshold=0.003):
        '''
            input:
                query:       B, M, 3
            output:
                mean_distance:  B, M
                penetrate_mask: B, M
        '''
        B, M, _ = query.shape
        query = query.reshape(-1, 3) * self.normalization_scale
        latent_inputs = self.latent_code.expand(query.shape[0], -1)
        inputs = torch.cat([latent_inputs, query], 1)
        pred_sdf = self.SDFDecoder(inputs).reshape(B, M) # 1

        pred_sdf /= self.normalization_scale
        mean_distance = pred_sdf.abs()
        penetrate_mask = pred_sdf < - threshold
        return mean_distance, penetrate_mask, pred_sdf