'''
The official implementation is in https://github.com/hassony2/manopth.
We change the default hand wrist to the origin, and speed up the code a little by decouping the computation of beta.
'''

import torch
from torch.nn import Module
import os 
import numpy as np
import chumpy as ch
import cv2
import pickle


class Rodrigues(ch.Ch):
    dterms = 'rt'

    def compute_r(self):
        return cv2.Rodrigues(self.rt.r)[0]

    def compute_dr_wrt(self, wrt):
        if wrt is self.rt:
            return cv2.Rodrigues(self.rt.r)[1].T


def lrotmin(p):
    if isinstance(p, np.ndarray):
        p = p.ravel()[3:]
        return np.concatenate(
            [(cv2.Rodrigues(np.array(pp))[0] - np.eye(3)).ravel()
            for pp in p.reshape((-1, 3))]).ravel()
    if p.ndim != 2 or p.shape[1] != 3:
        p = p.reshape((-1, 3))
    p = p[1:]
    return ch.concatenate([(Rodrigues(pp) - ch.eye(3)).ravel()
                        for pp in p]).ravel()


def ready_arguments(fname_or_dict, posekey4vposed='pose'):

    if not isinstance(fname_or_dict, dict):
        dd = pickle.load(open(fname_or_dict, 'rb'), encoding='latin1')
        # dd = pickle.load(open(fname_or_dict, 'rb'))
    else:
        dd = fname_or_dict

    want_shapemodel = 'shapedirs' in dd
    nposeparms = dd['kintree_table'].shape[1] * 3

    if 'trans' not in dd:
        dd['trans'] = np.zeros(3)
    if 'pose' not in dd:
        dd['pose'] = np.zeros(nposeparms)
    if 'shapedirs' in dd and 'betas' not in dd:
        dd['betas'] = np.zeros(dd['shapedirs'].shape[-1])

    for s in [
            'v_template', 'weights', 'posedirs', 'pose', 'trans', 'shapedirs',
            'betas', 'J'
    ]:
        if (s in dd) and not hasattr(dd[s], 'dterms'):
            dd[s] = ch.array(dd[s])

    assert (posekey4vposed in dd)

    pose_map_res = lrotmin(dd[posekey4vposed])
    dd_add = dd['posedirs'].dot(pose_map_res)
    dd['v_posed'] = dd['v_template'] + dd_add

    return dd

def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [batch_size, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [batch_size, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:,
                                                            2], norm_quat[:,
                                                                        3]

    batch_size = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([
        w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy,
        w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz,
        w2 - x2 - y2 + z2
    ],
                        dim=1).view(batch_size, 3, 3)
    return rotMat

def batch_rodrigues(axisang):
    #axisang N x 3
    axisang_norm = torch.norm(axisang + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(axisang_norm, -1)
    axisang_normalized = torch.div(axisang, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * axisang_normalized], dim=1)
    rot_mat = quat2mat(quat)
    rot_mat = rot_mat.view(rot_mat.shape[0], 9)
    return rot_mat

def th_posemap_axisang(pose_vectors):
    rot_nb = int(pose_vectors.shape[1] / 3)
    pose_vec_reshaped = pose_vectors.contiguous().view(-1, 3)
    rot_mats = batch_rodrigues(pose_vec_reshaped)
    rot_mats = rot_mats.view(pose_vectors.shape[0], rot_nb * 9)
    pose_maps = subtract_flat_id(rot_mats)
    return pose_maps, rot_mats


def th_with_zeros(tensor):
    batch_size = tensor.shape[0]
    padding = tensor.new([0.0, 0.0, 0.0, 1.0])
    padding.requires_grad = False

    concat_list = [tensor, padding.view(1, 1, 4).repeat(batch_size, 1, 1)]
    cat_res = torch.cat(concat_list, 1)
    return cat_res


def th_pack(tensor):
    batch_size = tensor.shape[0]
    padding = tensor.new_zeros((batch_size, 4, 3))
    padding.requires_grad = False
    pack_list = [padding, tensor]
    pack_res = torch.cat(pack_list, 2)
    return pack_res


def subtract_flat_id(rot_mats):
    # Subtracts identity as a flattened tensor
    rot_nb = int(rot_mats.shape[1] / 9)
    id_flat = torch.eye(
        3, dtype=rot_mats.dtype, device=rot_mats.device).view(1, 9).repeat(
            rot_mats.shape[0], rot_nb)
    # id_flat.requires_grad = False
    results = rot_mats - id_flat
    return results

class OurManoLayer(Module):
    __constants__ = [
        'rot', 'ncomps', 'ncomps', 'kintree_parents', 'check',
        'side', 'center_idx', 
    ]

    def __init__(self,
                side='right',
                mano_root='third_party/mano/models',
                ):
        super().__init__()
        self.rot = 3
        self.side = side
        self.ncomps = 45
        if side == 'right':
            self.mano_path = os.path.join(mano_root, 'MANO_RIGHT.pkl')
        elif side == 'left':
            self.mano_path = os.path.join(mano_root, 'MANO_LEFT.pkl')

        smpl_data = ready_arguments(self.mano_path)

        hands_components = smpl_data['hands_components']

        self.smpl_data = smpl_data

        self.register_buffer('th_betas',
                             torch.Tensor(smpl_data['betas'].r).unsqueeze(0))
        self.register_buffer('th_shapedirs',
                             torch.Tensor(smpl_data['shapedirs'].r))
        self.register_buffer('th_posedirs',
                             torch.Tensor(smpl_data['posedirs'].r))
        self.register_buffer(
            'th_v_template',
            torch.Tensor(smpl_data['v_template'].r).unsqueeze(0))
        self.register_buffer(
            'th_J_regressor',
            torch.Tensor(np.array(smpl_data['J_regressor'].toarray())))
        self.register_buffer('th_weights',
                             torch.Tensor(smpl_data['weights'].r))
        self.register_buffer('th_faces',
                             torch.Tensor(smpl_data['f'].astype(np.int32)).long())

        # Get hand mean
        hands_mean = np.zeros(hands_components.shape[1]) 
        th_hands_mean = torch.Tensor(hands_mean).unsqueeze(0)

        # Save as axis-angle
        self.register_buffer('th_hands_mean', th_hands_mean)
        selected_components = hands_components[:self.ncomps]
        self.register_buffer('th_comps', torch.Tensor(hands_components))
        self.register_buffer('th_selected_comps',
                                torch.Tensor(selected_components))

        # Kinematic chain params
        self.kintree_table = smpl_data['kintree_table']
        parents = list(self.kintree_table[0].tolist())
        self.kintree_parents = parents

    def pca_comps2pose(self,ncomps, pca):
        return pca.mm(self.th_comps[:ncomps])

    def register_beta(self, th_betas=torch.zeros(1)):
        self.th_v_shaped = torch.matmul(self.th_shapedirs,
                                       th_betas.transpose(1, 0)).permute(
                                           2, 0, 1) + self.th_v_template
        self.th_j = torch.matmul(self.th_J_regressor, self.th_v_shaped)
        return 

    def forward(self,
                th_pose_coeffs,
                th_betas=torch.zeros(1),
                th_trans=torch.zeros(1),
                root_palm=torch.Tensor([0]),
                share_betas=torch.Tensor([0]),
                original_version=False,
                use_registed_beta=False
                ):
        """
        Args:
        th_trans (Tensor (batch_size x ncomps)): if provided, applies trans to joints and vertices
        th_betas (Tensor (batch_size x 10)): if provided, uses given shape parameters for hand shape
        else centers on root joint (9th joint)
        root_palm: return palm as hand root instead of wrist
        original_version: use the official version, which means the wrist is not at the origin.
        """
        # if len(th_pose_coeffs) == 0:
        #     return th_pose_coeffs.new_empty(0), th_pose_coeffs.new_empty(0)

        batch_size = th_pose_coeffs.shape[0]
        # Get axis angle from PCA components and coefficients
        # Remove global rot coeffs
        th_hand_pose_coeffs = th_pose_coeffs[:, self.rot:self.rot +
                                                self.ncomps]
        th_full_hand_pose = th_hand_pose_coeffs

        # Concatenate back global rot
        th_full_pose = torch.cat([
            th_pose_coeffs[:, :self.rot],
            self.th_hands_mean + th_full_hand_pose
        ], 1)

        # compute rotation matrixes from axis-angle while skipping global rotation
        th_pose_map, th_rot_map = th_posemap_axisang(th_full_pose)
        root_rot = th_rot_map[:, :9].view(batch_size, 3, 3)
        th_rot_map = th_rot_map[:, 9:]
        th_pose_map = th_pose_map[:, 9:]
   
        if use_registed_beta:
            th_v_shaped = self.th_v_shaped
            th_j = self.th_j.repeat(batch_size, 1, 1)
        else:
            # Full axis angle representation with root joint
            if th_betas is None or th_betas.numel() == 1:
                th_v_shaped = torch.matmul(self.th_shapedirs,
                                        self.th_betas.transpose(1, 0)).permute(
                                            2, 0, 1) + self.th_v_template
                th_j = torch.matmul(self.th_J_regressor, th_v_shaped).repeat(
                    batch_size, 1, 1)
            else:
                if share_betas:
                    th_betas = th_betas.mean(0, keepdim=True).expand(th_betas.shape[0], 10)
                th_v_shaped = torch.matmul(self.th_shapedirs,
                                        th_betas.transpose(1, 0)).permute(
                                            2, 0, 1) + self.th_v_template
                th_j = torch.matmul(self.th_J_regressor, th_v_shaped)

                # th_pose_map should have shape 20x135
        th_v_posed = th_v_shaped + torch.matmul(
            self.th_posedirs, th_pose_map.transpose(0, 1)).permute(2, 0, 1)
        # Final T pose with transformation done !

        # Global rigid transformation
        
        root_j = th_j[:, 0, :].contiguous().view(batch_size, 3, 1)
        root_trans = th_with_zeros(torch.cat([root_rot, root_j], 2))

        all_rots = th_rot_map.view(th_rot_map.shape[0], 15, 3, 3)
        lev1_idxs = [1, 4, 7, 10, 13]
        lev2_idxs = [2, 5, 8, 11, 14]
        lev3_idxs = [3, 6, 9, 12, 15]
        lev1_rots = all_rots[:, [idx - 1 for idx in lev1_idxs]]
        lev2_rots = all_rots[:, [idx - 1 for idx in lev2_idxs]]
        lev3_rots = all_rots[:, [idx - 1 for idx in lev3_idxs]]
        lev1_j = th_j[:, lev1_idxs]
        lev2_j = th_j[:, lev2_idxs]
        lev3_j = th_j[:, lev3_idxs]

        # From base to tips
        # Get lev1 results
        all_transforms = [root_trans.unsqueeze(1)]
        lev1_j_rel = lev1_j - root_j.transpose(1, 2)
        lev1_rel_transform_flt = th_with_zeros(torch.cat([lev1_rots, lev1_j_rel.unsqueeze(3)], 3).view(-1, 3, 4))
        root_trans_flt = root_trans.unsqueeze(1).repeat(1, 5, 1, 1).view(root_trans.shape[0] * 5, 4, 4)
        lev1_flt = torch.matmul(root_trans_flt, lev1_rel_transform_flt)
        all_transforms.append(lev1_flt.view(all_rots.shape[0], 5, 4, 4))

        # Get lev2 results
        lev2_j_rel = lev2_j - lev1_j
        lev2_rel_transform_flt = th_with_zeros(torch.cat([lev2_rots, lev2_j_rel.unsqueeze(3)], 3).view(-1, 3, 4))
        lev2_flt = torch.matmul(lev1_flt, lev2_rel_transform_flt)
        all_transforms.append(lev2_flt.view(all_rots.shape[0], 5, 4, 4))

        # Get lev3 results
        lev3_j_rel = lev3_j - lev2_j
        lev3_rel_transform_flt = th_with_zeros(torch.cat([lev3_rots, lev3_j_rel.unsqueeze(3)], 3).view(-1, 3, 4))
        lev3_flt = torch.matmul(lev2_flt, lev3_rel_transform_flt)
        all_transforms.append(lev3_flt.view(all_rots.shape[0], 5, 4, 4))

        reorder_idxs = [0, 1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 14, 5, 10, 15]
        th_results = torch.cat(all_transforms, 1)[:, reorder_idxs]
        th_results_global = th_results

        joint_js = torch.cat([th_j, th_j.new_zeros(th_j.shape[0], 16, 1)], 2)
        tmp2 = torch.matmul(th_results, joint_js.unsqueeze(3))
        th_results2 = (th_results - torch.cat([tmp2.new_zeros(*tmp2.shape[:2], 4, 3), tmp2], 3)).permute(0, 2, 3, 1)

        th_T = torch.matmul(th_results2, self.th_weights.transpose(0, 1))

        th_rest_shape_h = torch.cat([
            th_v_posed.transpose(2, 1),
            torch.ones((batch_size, 1, th_v_posed.shape[1]),
                       dtype=th_T.dtype,
                       device=th_T.device),
        ], 1)

        th_verts = (th_T * th_rest_shape_h.unsqueeze(1)).sum(2).transpose(2, 1)
        th_verts = th_verts[:, :, :3]
        th_jtr = th_results_global[:, :, :3, 3]
        # In addition to MANO reference joints we sample vertices on each finger
        # to serve as finger tips
        if self.side == 'right':
            tips = th_verts[:, [745, 317, 444, 556, 673]]
        else:
            tips = th_verts[:, [745, 317, 445, 556, 673]]
        if bool(root_palm):
            palm = (th_verts[:, 95] + th_verts[:, 22]).unsqueeze(1) / 2
            th_jtr = torch.cat([palm, th_jtr[:, 1:]], 1)
        th_jtr = torch.cat([th_jtr, tips], 1)

        # Reorder joints to match visualization utilities
        th_jtr = th_jtr[:, [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]]
        if not original_version:
            center_joint = th_jtr[:, 0].unsqueeze(1)
            th_jtr = th_jtr - center_joint
            th_verts = th_verts - center_joint
    
        th_jtr = th_jtr + th_trans.unsqueeze(1)
        th_verts = th_verts + th_trans.unsqueeze(1)

        # return th_verts, th_jtr, center_joint       # in miles
        return th_verts, th_jtr       # in miles



if __name__ == '__main__':
    mm = OurManoLayer()

    B = 1
    th_pose_coeffs = torch.randn((B,48))
    trans = torch.randn((B,3))
    rand_beta = torch.zeros((B,10))
    from copy import deepcopy
    tt = deepcopy(trans)
    hand, kp, tem = mm.forward(
                    th_pose_coeffs=th_pose_coeffs, 
                    th_trans=trans,
                    th_betas=rand_beta)
    theta = th_pose_coeffs[:,:3].norm(dim=-1)
    axis = th_pose_coeffs[:,:3] / theta
    from manopth.manolayer import ManoLayer
    
    mano_layer_right = ManoLayer(
                mano_root='/home/jiayichen/manopth/mano/models', side='right', use_pca=False, ncomps=45)

    hand1, kp1 = mano_layer_right.forward(
                    th_pose_coeffs=th_pose_coeffs, 
                    th_trans=trans,
                    th_betas=rand_beta)

    print('compare with formal mano:', kp1/1000-kp-tem)

    th_pose_coeffs[:,:3] = 0
    trans[:,:3] = 0
    hand2, kp2,_ = mm.forward(
                    th_pose_coeffs=th_pose_coeffs, 
                    th_trans=trans,
                    th_betas=rand_beta)

    import sys
    sys.path.append('../..')
    from pose_utils.rotations import axis_theta_to_matrix
    
    r = axis_theta_to_matrix(axis,theta)[0]

    kp2 = torch.matmul(kp2, r.transpose(-1,-2))+tt
    print('check R, T: ', kp2-kp)

    