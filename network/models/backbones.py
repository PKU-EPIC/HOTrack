import os
import sys
from os.path import join as pjoin
BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, pjoin(BASEPATH, '..'))
sys.path.insert(0, pjoin(BASEPATH, '..', '..'))
import argparse
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.jit as jit

from pointnet_utils import PointNetSetAbstraction, PointNetFeaturePropagation, PointNetSetAbstractionMsg,\
    PointNetSetAbstractionMsg_fast, PointNetSetAbstraction_fast, PointNetFeaturePropagation_fast


class PointNet2Msg(nn.Module):
    """
    Output: 128 channels
    """
    def __init__(self, cfg, out_dim, net_type='camera', use_xyz_feat=False, init_feature_dim=0):
        super(PointNet2Msg, self).__init__()
        net_cfg = cfg['pointnet'][net_type]
        self.out_dim = out_dim
        self.in_dim = init_feature_dim+3 if use_xyz_feat else init_feature_dim
        self.use_xyz_feat = use_xyz_feat
        self.sa1 = PointNetSetAbstractionMsg(npoint=net_cfg['sa1']['npoint'],
                                             radius_list=net_cfg['sa1']['radius_list'],
                                             nsample_list=net_cfg['sa1']['nsample_list'],
                                             in_channel=self.in_dim + 3,
                                             mlp_list=net_cfg['sa1']['mlp_list'])

        self.sa2 = PointNetSetAbstractionMsg(npoint=net_cfg['sa2']['npoint'],
                                             radius_list=net_cfg['sa2']['radius_list'],
                                             nsample_list=net_cfg['sa2']['nsample_list'],
                                             in_channel=self.sa1.out_channel + 3,
                                             mlp_list=net_cfg['sa2']['mlp_list'])

        self.sa3 = PointNetSetAbstraction(npoint=None,
                                          radius=None,
                                          nsample=None,
                                          in_channel=self.sa2.out_channel + 3,
                                          mlp=net_cfg['sa3']['mlp'], group_all=True)

        self.fp3 = PointNetFeaturePropagation(in_channel=(self.sa2.out_channel + self.sa3.out_channel),
                                              mlp=net_cfg['fp3']['mlp'])
        self.fp2 = PointNetFeaturePropagation(in_channel=(self.sa1.out_channel + self.fp3.out_channel),
                                              mlp=net_cfg['fp2']['mlp'])
        self.fp1 = PointNetFeaturePropagation(in_channel=(self.in_dim + 3 + self.fp2.out_channel),
                                              mlp=net_cfg['fp1']['mlp'])

        self.conv1 = nn.Conv1d(self.fp1.out_channel, self.out_dim, 1)
        self.bn1 = nn.BatchNorm1d(self.out_dim)

        self.device = cfg['device']

    def forward(self, input):  # [B, 3, N]
        l0_xyz = input[:, :3]
        if self.use_xyz_feat:
            l0_points = input
        else:
            l0_points = input[:, 3:]
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points], dim=1), l1_points)

        feat = F.relu(self.bn1(self.conv1(l0_points)))
        return feat

class PointNet2Msg_fast(nn.Module):
    """
    We don't know why but in our experiments we find that this module is much faster than PointNet2Msg, with exactly the same behaviour.
    """
    def __init__(self, cfg, out_dim, net_type='camera', use_xyz_feat=False, init_feature_dim=0):
        super(PointNet2Msg_fast, self).__init__()
        net_cfg = cfg['pointnet'][net_type]
        self.out_dim = out_dim
        self.in_dim = init_feature_dim + 3 if use_xyz_feat else init_feature_dim
        self.use_xyz_feat = use_xyz_feat
        self.sa1 = PointNetSetAbstractionMsg_fast(npoint=net_cfg['sa1']['npoint'],
                                             radius_list=net_cfg['sa1']['radius_list'],
                                             nsample_list=net_cfg['sa1']['nsample_list'],
                                             in_channel=self.in_dim + 3,
                                             mlp_list=net_cfg['sa1']['mlp_list'])

        self.sa2 = PointNetSetAbstractionMsg_fast(npoint=net_cfg['sa2']['npoint'],
                                             radius_list=net_cfg['sa2']['radius_list'],
                                             nsample_list=net_cfg['sa2']['nsample_list'],
                                             in_channel=self.sa1.out_channel + 3,
                                             mlp_list=net_cfg['sa2']['mlp_list'])

        self.sa3 = PointNetSetAbstraction_fast(npoint=None,
                                          radius=None,
                                          nsample=None,
                                          in_channel=self.sa2.out_channel + 3,
                                          mlp=net_cfg['sa3']['mlp'], group_all=True)

        self.fp3 = PointNetFeaturePropagation_fast(in_channel=(self.sa2.out_channel + self.sa3.out_channel),
                                              mlp=net_cfg['fp3']['mlp'])
        self.fp2 = PointNetFeaturePropagation_fast(in_channel=(self.sa1.out_channel + self.fp3.out_channel),
                                              mlp=net_cfg['fp2']['mlp'])
        self.fp1 = PointNetFeaturePropagation_fast(in_channel=(self.in_dim + 3 + self.fp2.out_channel),
                                              mlp=net_cfg['fp1']['mlp'])

        self.conv1 = nn.Conv1d(self.fp1.out_channel, self.out_dim, 1)
        self.bn1 = nn.BatchNorm1d(self.out_dim)

        self.device = cfg['device']

    def forward(self, input):  # [B, P, 3 + 3, N]
        B, C, N = input.shape
        input = input.reshape(B, 1, C, N)
        l0_xyz = input[:, :, :3] # B, P, 3, N
        if self.use_xyz_feat:
            l0_points = input
        else:
            l0_points = input[:, :, 3:]
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        if l0_points.shape[-2]:
            l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points], dim=-2), l1_points)
        else:
            l0_points = self.fp1(l0_xyz, l1_xyz, l0_xyz, l1_points)
        l0_points = l0_points.reshape(B, -1, N)
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        return feat

class PointNet2Encoder(nn.Module):
    """
    Output: 128 channels
    """
    def __init__(self, cfg, out_dim, net_type='camera', use_xyz_feat=False, use_init_label=False, use_one_hot=False):
        super(PointNet2Encoder, self).__init__()
        net_cfg = cfg['pointnet'][net_type]
        self.out_dim = out_dim
        self.in_dim = 3 if use_xyz_feat else 0
        self.use_xyz_feat = use_xyz_feat
        if use_init_label:
            self.in_dim += 1
        if use_one_hot:
            self.in_dim += 22
        self.sa1 = PointNetSetAbstractionMsg(npoint=net_cfg['sa1']['npoint'],
                                             radius_list=net_cfg['sa1']['radius_list'],
                                             nsample_list=net_cfg['sa1']['nsample_list'],
                                             in_channel=self.in_dim + 3,
                                             mlp_list=net_cfg['sa1']['mlp_list'])

        self.sa2 = PointNetSetAbstractionMsg(npoint=net_cfg['sa2']['npoint'],
                                             radius_list=net_cfg['sa2']['radius_list'],
                                             nsample_list=net_cfg['sa2']['nsample_list'],
                                             in_channel=self.sa1.out_channel + 3,
                                             mlp_list=net_cfg['sa2']['mlp_list'])

        self.sa3 = PointNetSetAbstraction(npoint=None,
                                          radius=None,
                                          nsample=None,
                                          in_channel=self.sa2.out_channel + 3,
                                          mlp=net_cfg['sa3']['mlp'], group_all=True)

        self.conv1 = nn.Conv1d(self.sa3.out_channel, 256, 1)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(256, self.out_dim, 1)
        self.bn2 = nn.BatchNorm1d(self.out_dim)

        self.device = cfg['device']

    def forward(self, input):  # [B, 3, N]
        l0_xyz = input[:, :3]
        if self.use_xyz_feat:
            l0_points = input
        else:
            l0_points = input[:, 3:]
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        feat = self.drop1(F.relu(self.bn1(self.conv1(l3_points))))
        feat = F.relu(self.bn2(self.conv2(feat)))
        return feat

def parse_args():
    parser = argparse.ArgumentParser('SingleFrameModel')
    parser.add_argument('--config', type=str, default='config.yml', help='path to config.yml')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    from configs.config import get_config
    cfg = get_config(args, save=False)
    msg = PointNet2Msg(cfg, out_dim=128)
    print(msg)


