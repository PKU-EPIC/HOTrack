import sys
import os
from os.path import join as pjoin
import torch 
from third_party.mano.our_mano import OurManoLayer

BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, BASEPATH)
sys.path.insert(0, pjoin(BASEPATH, '..'))
sys.path.insert(0, pjoin(BASEPATH, '..', '..'))
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer import PositionEmbeddingSine, attn_module, TransT
from backbones import PointNet2Msg_fast
from blocks import rearrange_module
from pointnet_utils import PointNetSetAbstractionMsg_GivenCenterPoints
import numpy as np
from hand_utils import *
from pose_utils.rotations import matrix_to_unit_quaternion

def L2_loss(x, y, mask=None):  # x,y:[B,3,x], mask:[B, 1, x]
    assert x.shape[1] == 3
    assert y.shape[1] == 3
    assert mask is None or mask.shape[1] == 1

    if mask is None:
        return (x - y).norm(dim=1).mean()
    else:
        return (((x - y) * mask).norm(dim=1).sum(dim=-1) / torch.clamp(mask.sum(dim=-1), min=1).squeeze()).mean()


def L1_loss(x, y, mask=None, check_dim_in=3):  # x,y: [B,3,x], mask: [B, 1, x]
    assert x.shape[1] == check_dim_in
    assert y.shape[1] == check_dim_in
    assert mask is None or mask.shape[1] == 1

    if mask is None:
        return (x - y).abs().mean()
    else:
        return (((x - y) * mask).abs().mean(dim=1).sum(dim=-1) / torch.clamp(mask.sum(dim=-1), min=1).squeeze()).mean()


class HandTrackNet(nn.Module):
    def __init__(self, cfg):
        super(HandTrackNet, self).__init__()
        self.error = []
        for i in range(51):
            self.error.append([])
        self.device = cfg['device']
        self.handframe = cfg['network']['handframe']
        pointnet_outchannel = cfg['network']['backbone_out_dim']
        self.bhand = PointNet2Msg_fast(cfg, pointnet_outchannel)
        self.r1 = rearrange_module(channel=pointnet_outchannel)
        self.r2 = rearrange_module(channel=pointnet_outchannel)

        assert pointnet_outchannel % 6 == 0
        self.positionEmbedding = PositionEmbeddingSine(num_pos_feats=pointnet_outchannel // 6)

        self.q1 = PointNetSetAbstractionMsg_GivenCenterPoints(radius_list=[0.2, 0.2], nsample_list=[16, 64],
                                                              mlp_list=[[128, 128, pointnet_outchannel // 2], [128, 128, pointnet_outchannel // 2]],
                                                              in_channel=pointnet_outchannel + 3,
                                                              knn=True)
       
        self.q2 = PointNetSetAbstractionMsg_GivenCenterPoints(radius_list=[0.2, 0.2], nsample_list=[16, 64],
                                                              mlp_list=[[128, 128, pointnet_outchannel // 2], [128, 128, pointnet_outchannel // 2]],
                                                              in_channel=pointnet_outchannel * 2 + 3,
                                                              knn=True)
        
        self.transt = TransT(d_model=pointnet_outchannel)
        self.c3 = attn_module(d_model=pointnet_outchannel)

        self.final_mlp = nn.Sequential(nn.Conv1d(pointnet_outchannel, 256, 1),
                                           nn.ReLU(inplace=True),
                                           nn.Conv1d(256, 3, 1))
        
    def forward(self, input, flag_dict):
        '''
        input: jittered_hand_kp [B, kp_num, 3]
               points [B, N, 3]
        output: pred_hand_kp [B, kp_num, 3]
        '''
        track_flag = flag_dict['track_flag']
        # --------------------- data prepare ----------------------
        if track_flag:
            palm_template = input['pred_palm_template']
        else:
            palm_template = input['gt_hand_pose']['palm_template'].to(self.device)

        jittered_kp = input['jittered_hand_kp'].to(self.device).float() # [B,kp_num,3]
        hand_points = input['hand_points'].to(self.device).float()  # [B,N,3]
        ret_dict = {}

        # --------------------- compute hand frame -------------------
        
        if self.handframe == 'kp':
            canon_pose = {}
            canon_pose['scale'] = 0.2 * torch.ones(1, device=self.device).float()
            canon_pose['rotation'], canon_pose['translation'], _, _, _ = ransac_rt(palm_template, handkp2palmkp(jittered_kp))
            ret_dict['canon_pose'] = canon_pose
        elif self.handframe == 'OBB':
            canon_pose = input['OBB_pose']
            for key in canon_pose:
                canon_pose[key] = canon_pose[key].to(self.device).float()
            ret_dict['canon_pose'] = canon_pose
        elif self.handframe == 'camera':
            b = hand_points.shape[0]
            canon_pose = {}
            canon_pose['scale'] = 0.2* torch.ones(1, device=self.device).float()
            canon_pose['rotation'] = torch.eye(3, device=self.device).float().unsqueeze(0).repeat(b,1,1)
            canon_pose['translation'] = torch.zeros((b, 3, 1), device=self.device).float()
            ret_dict['canon_pose'] = canon_pose
        else:
            raise NotImplementedError

        # -------------------- canonicalize ---------------------------
        cam = torch.cat([hand_points, jittered_kp], dim=1).transpose(2, 1)  # [B, 3, N]
        cam = canonicalize(cam, canon_pose)

        # ----------------------- network forward -----------------------
        position_embedding = self.positionEmbedding(cam)
        kp_num = jittered_kp.shape[1]
        pos2 = position_embedding[..., :-kp_num]        # no use
        pos1 = position_embedding[..., -kp_num:]        # no use
        xyz2 = cam[..., :-kp_num]
        xyz1 = cam[..., -kp_num:]

        # backbone
        src2 = self.bhand(xyz2)

        f11, pre_group_idx = self.q1(xyz2, src2, xyz1, None, return_group_idx=True)
        f12 = self.r1(f11)
        f13 = self.q2(xyz2, src2, xyz1, f12, pre_group_idx=pre_group_idx)
        f14 = self.r2(f13)
        
        # NOTE: We tried attention modules but we failed to make it work.
        # But we surprisingly find the MLP in self.transt can improve the performance...so we simply keep it.
        f15, f251 = self.transt(src1=f14, pos1=pos1, src2=src2, pos2=pos2, attn=False)
        fusioned_feature = self.c3(f15, pos1, f251, pos2, attn=False)

        ret_dict['pred_kp_handframe'] = self.final_mlp(fusioned_feature) + xyz1  # [B,3,kp_num]

        # ------------------------ post process ---------------------------
        ret_dict['init_kp_handframe'] = xyz1
        ret_dict['points_handframe'] = xyz2
        ret_dict['pred_kp'] = decanonicalize(ret_dict['pred_kp_handframe'], canon_pose).transpose(2, 1)

        if 'IKNet_flag' in flag_dict and flag_dict['IKNet_flag']:
            avg_dis_4nn, _ = knn_point(4, ret_dict['pred_kp'], hand_points)
            avg_dis_4nn = torch.mean(avg_dis_4nn, dim=-1)
            avg_dis_4nn[:, 0] -= 0.01
            avg_dis_4nn[:, 1] -= 0.01
            visibility = (avg_dis_4nn < 0.02).bool()
            ret_dict['pred_kp_vis_mask'] = visibility

        return ret_dict

    def compute_loss(self, input, ret_dict, flag_dict):
        '''
            ret_dict:
                canon_pose: hand frame to camera frame
                init_kp_handframe
                pred_kp_handframe
        '''
        # ------------------------ data prepare ---------------------------
        gt_kp = input['gt_hand_kp'].to(self.device).float().transpose(-1, -2)  # [B,kp_num,3]
        pred_kp = ret_dict['pred_kp'].transpose(-1, -2)
        canon_pose = ret_dict['canon_pose']
        gt_kp_handframe = canonicalize(gt_kp, canon_pose)
        ret_dict['gt_kp_handframe'] = gt_kp_handframe
        init_kp_scaled = ret_dict['init_kp_handframe'] * canon_pose['scale'][:, None, None]
        pred_kp_scaled = ret_dict['pred_kp_handframe'] * canon_pose['scale'][:, None, None]
        gt_kp_scaled = ret_dict['gt_kp_handframe'] * canon_pose['scale'][:, None, None]
        if self.handframe != 'OBB':
            if 'global_pose' in ret_dict:
                gt_delta_rotation = input['gt_hand_pose']['rotation'].to(self.device).float().reshape(-1, 3, 3)
                gt_delta_translation = input['gt_hand_pose']['translation'].to(self.device).float().reshape(-1, 3, 1)
                delta_rotation, delta_translation = ret_dict['global_pose']['rotation'].reshape(-1, 3, 3), ret_dict['global_pose']['translation'].reshape(-1, 3, 1)
            else:
                palm_template = input['gt_hand_pose']['palm_template'].to(self.device)
                gt_delta_rotation, gt_delta_translation, _, _, _ = ransac_rt(palm_template, handkp2palmkp(gt_kp_scaled.transpose(-1,-2)))
                delta_rotation, delta_translation, _, _, _ = ransac_rt(palm_template, handkp2palmkp(pred_kp_scaled.transpose(-1,-2)))

        # ------------------------- compute loss --------------------------
        loss_dict = {}
        loss_dict['hand_pred_kp_loss'] = L1_loss(pred_kp_scaled, gt_kp_scaled)
        loss_dict['hand_pred_kp_diff'] = L2_loss(pred_kp, gt_kp)
        loss_dict['hand_init_kp_diff'] = L2_loss(init_kp_scaled, gt_kp_scaled)

        if self.handframe != 'OBB':
            loss_dict['hand_pred_r_loss'] = L1_loss(delta_rotation, gt_delta_rotation)
            loss_dict['hand_pred_t_loss'] = L1_loss(delta_translation, gt_delta_translation)
            rot_err_mat = torch.matmul(delta_rotation.transpose(-1, -2), gt_delta_rotation)  # B, 3, 3
            if 'global_pose' not in ret_dict:
                loss_dict['hand_init_r_diff'] = torch.mean(torch.acos(
                            torch.clamp((gt_delta_rotation[:, 0, 0] + gt_delta_rotation[:, 1, 1]
                            + gt_delta_rotation[:, 2, 2] - 1) / 2, min=-1, max=1))) * 180 / np.pi
                loss_dict['hand_init_t_diff'] = gt_delta_translation.norm(dim=1).mean()
            loss_dict['hand_pred_r_diff'] = torch.mean(torch.acos(
                torch.clamp((rot_err_mat[:, 0, 0] + rot_err_mat[:, 1, 1] + rot_err_mat[:, 2, 2] - 1) / 2, min=-1,
                            max=1))) * 180 / np.pi
            loss_dict['hand_pred_t_diff'] = L2_loss(delta_translation, gt_delta_translation)
        

        if flag_dict['track_flag']:
            gt_rotation = input['gt_hand_pose']['rotation'].to(self.device).float().reshape(-1, 3, 3)
            gt_translation = input['gt_hand_pose']['translation'].to(self.device).float().reshape(-1, 3, 1)
            canon_rotation = canon_pose['rotation'].reshape(-1, 3, 3)
            canon_translation = canon_pose['translation'].reshape(-1, 3, 1)
            rot_err_mat = torch.matmul(canon_rotation.transpose(-1, -2), gt_rotation)  # B, 3, 3
            loss_dict['hand_canon_r_diff'] = torch.mean(torch.acos(
                torch.clamp((rot_err_mat[:, 0, 0] + rot_err_mat[:, 1, 1]
                            + rot_err_mat[:, 2, 2] - 1) / 2, min=-1, max=1))) * 180 / np.pi
            loss_dict['hand_canon_t_diff'] = L2_loss(gt_translation, canon_translation)
            
        if flag_dict['IKNet_flag']:
            gt_MANO_theta = input['gt_hand_pose']['mano_pose'][:, 3:].to(self.device).float()
            loss_dict['MANO_theta_diff'] = L1_loss(ret_dict['MANO_theta'], gt_MANO_theta, check_dim_in=45)
           
        return loss_dict, ret_dict

    def visualize(self, input, ret_dict, flag_dict):
        save_flag = flag_dict['save_flag']
        # ----------------------for debug------------------------------
        from vis_utils import hand_vis,plot3d_pts
        rand_num = np.random.randint(0, ret_dict['pred_kp_handframe'].shape[0])
        init_kp = ret_dict['init_kp_handframe'][rand_num].transpose(-1, -2).cpu()
        pred_kp = ret_dict['pred_kp_handframe'][rand_num].transpose(-1, -2).detach().cpu()
        gt_kp = ret_dict['gt_kp_handframe'][rand_num].transpose(-1, -2).cpu()
        points = ret_dict['points_handframe'][rand_num].transpose(-1, -2).cpu()

        if save_flag:
            save_name = input['file_name'][rand_num]
            print('save to ./debug/{}'.format(save_name))
           # hand_vis(points, init_kp, pred_kp, gt_kp, show_fig=False, save_fig=True, save_folder='./debug',
           #          save_name=save_name)
            plot3d_pts([[points, gt_kp],[points, pred_kp]],
                       show_fig=False, save_fig=True, save_folder='./debug', save_name=save_name)
        else:
            print('show figure!')
            hand_vis(points, init_kp, pred_kp, gt_kp, show_fig=True, save_fig=False)
        return


class IKNet(nn.Module):
    def __init__(self, cfg):
        super(IKNet, self).__init__()
        self.device = cfg['device']
        self.linear = nn.ModuleList()
        self.bn = nn.ModuleList()
        last_dim = 21 * 3 * 2
        weight = 1024
        self.layer_num = 6
        for i in range(self.layer_num):
            self.linear.append(nn.Linear(last_dim, weight))
            self.bn.append(nn.BatchNorm1d(weight))
            last_dim = weight
        self.linear.append(nn.Linear(weight, 15 * 4))
        self.iknetframe = cfg['network']['iknetframe']
        self.mano_layer_right = OurManoLayer(mano_root=cfg['mano_root'], side='right').cuda()


    def forward(self, input, flag_dict):
        ret_dict = {}
        track_flag = flag_dict['track_flag']
        b = input['gt_hand_kp'].shape[0]
        if not track_flag:
            palm_template = input['gt_hand_pose']['palm_template'].to(self.device)
            init_kp = input['jittered_hand_kp'].to(self.device)
            beta = input['gt_hand_pose']['mano_beta'].to(self.device)
        else:
            palm_template = input['pred_palm_template']
            init_kp = input['baseline_pred_kp'].to(self.device)
            beta = input['pred_beta']
        
        canon_pose = {}
        canon_pose['scale'] = 0.2 * torch.ones(1, device=self.device).float()
        canon_pose['rotation'], canon_pose['translation'], _, _, _ = ransac_rt(palm_template, handkp2palmkp(init_kp))
        
        if self.iknetframe == 'kp':
            # canonicalize to hand frame
            init_kp_handframe = canonicalize(init_kp.transpose(-1,-2), canon_pose)   
        elif self.iknetframe == 'camera':
            init_kp_handframe = init_kp.transpose(-1,-2) * 5
        else:
            raise NotImplementedError

        # compute bone 
        parent_index = [0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
        init_bone = init_kp_handframe - init_kp_handframe[..., parent_index]
        pack = torch.cat([init_kp_handframe.reshape(b, -1), init_bone.reshape(b, -1)], -1)
        ret_dict['init_kp_handframe'] = init_kp_handframe
        ret_dict['init_kp'] = init_kp
        if not track_flag:
            if self.iknetframe == 'kp':
                ret_dict['gt_kp_handframe'] = canonicalize(input['gt_hand_kp'].to(self.device).transpose(-1,-2), canon_pose)   
            elif self.iknetframe == 'camera':
                ret_dict['gt_kp_handframe'] = input['gt_hand_kp'].to(self.device).transpose(-1,-2) * 5

        # --------------------network-----------------
        for i in range(self.layer_num):
            pack = F.relu(self.bn[i](self.linear[i](pack)))
        raw_quat = self.linear[self.layer_num](pack)

        # --------------------post process--------------
        ret_dict['raw_quat'] = raw_quat

        # if not track_flag:
        gt_mano_pose = input['gt_hand_pose']['mano_pose'].float().to(self.device)
        anno_quat = mano_axisang2quat(gt_mano_pose)
        ret_dict['gt_quat'] = anno_quat[:, 4:]
        if track_flag and not flag_dict['opt_flag']:
            # use estimated global R, T                                              
            _, pred_kp = self.mano_layer_right.forward(
                th_pose_coeffs=mano_quat2axisang(torch.cat([matrix_to_unit_quaternion(canon_pose['rotation']), raw_quat], dim=1)),
                th_trans=canon_pose['translation'].reshape(b, 3), th_betas=beta.reshape(b, 10).float())   
            ret_dict['pred_kp'] = pred_kp

        ret_dict['MANO_theta'] = mano_quat2axisang(raw_quat) #B, 45
        ret_dict['global_pose'] = canon_pose
        return ret_dict

    def compute_loss(self, data, ret_dict, flag_dict):
        gt_kp = data['gt_hand_kp'].to(self.device).transpose(-1, -2) #[b, 3, kp_num]
        # pred_kp = ret_dict['pred_kp'].transpose(-1,-2)
        # no_global_pred_kp = ret_dict['no_global_pred_kp'].transpose(-1,-2)
        init_kp = ret_dict['init_kp'].transpose(-1,-2)

        loss_dict = {}
        # only supervise the rotation and position of each joints
        loss_dict['quat_loss'] = (ret_dict['raw_quat'] - ret_dict['gt_quat']).abs().mean()
        loss_dict['init_gt_kp_diff'] =  L2_loss(init_kp, gt_kp)

        return loss_dict, ret_dict

    def visualize(self, input, ret_dict, flag_dict):
        from vis_utils import plot3d_pts
        save_flag = flag_dict['save_flag']
        rand_num = np.random.randint(0, ret_dict['pred_kp'].shape[0])
        init_kp_handframe = ret_dict['init_kp_handframe'][rand_num].cpu().detach().transpose(-1,-2)
        gt_kp_handframe = ret_dict['gt_kp_handframe'][rand_num].cpu().detach().transpose(-1,-2)
        pred_kp_handframe = ret_dict['pred_kp_handframe'][rand_num].cpu().detach().transpose(-1,-2)

        if save_flag:
            save_name = input['file_name'][rand_num]
            print('save to ./debug/{}'.format(save_name))
            plot3d_pts([[gt_kp_handframe,init_kp_handframe], [gt_kp_handframe,pred_kp_handframe]],
                       show_fig=False, save_fig=True, save_folder='./debug', save_name='invis_' + save_name)
        else:
            print('show figure!')
        return

def parse_args():
    parser = argparse.ArgumentParser('SingleFrameModel')
    parser.add_argument('--config', type=str, default='1.17_MANO_LSTM.yml', help='path to config.yml')
    return parser.parse_args()


if __name__ == '__main__':
    from configs.config import get_config
    args = parse_args()
    cfg = get_config(args, save=False)

