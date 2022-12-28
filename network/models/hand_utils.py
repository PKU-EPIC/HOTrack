import torch
import numpy as np
from pointnet_utils import knn_point, group_operation
import sys
sys.path.append('../..')
from pose_utils.rotations import axis_theta_to_quater, quater_to_axis_theta
from os.path import join as pjoin
import os 
epison = 1e-7
BASEPATH = os.path.dirname(__file__)


def mano_quat2axisang(quat):
    axisang = []
    for i in range(quat.shape[1]//4):
        axis, theta = quater_to_axis_theta(quat[:, i * 4:i * 4 + 4])
        axisang.append(axis * theta[:, None])
    axisang = torch.cat(axisang, dim=1)
    return axisang

def mano_axisang2quat(axisang):
    q = []
    for i in range(axisang.shape[1] // 3):
        theta = axisang[:, i * 3:i * 3 + 3].norm(dim=-1,keepdim=True)
        axis = axisang[:, i * 3:i * 3 + 3] / (theta + epison)
        q.append(axis_theta_to_quater(axis, theta.squeeze(-1)))
    q = torch.cat(q, dim=1)
    return q

def canonicalize(data, canon_pose):  # data: [B, 3, N]
    return torch.matmul(canon_pose['rotation'].transpose(-1, -2),
                        data - canon_pose['translation']) / canon_pose['scale'][:, None, None]

def decanonicalize(data, canon_pose):  # data: [B, 3, N]
    return canon_pose['scale'][:, None, None] * torch.matmul(canon_pose['rotation'],
                                                             data) + canon_pose['translation']

def normalize(q):
    norm = torch.norm(q, dim=-1, keepdim=True)
    return q / (norm + epison)

def solve_rot_and_trans(x, y, cpu=True):
    """
    Solve R and t, such that y = R @ x + t
    convert to cpu to speed up!
    :param x: [B, num, 3]
    :param y: [B, num, 3]
    :return R: [B, 3, 3]
    :return t: [B, 3, 1]
    """
    cx, cy = x.mean(dim=1, keepdim=True), y.mean(dim=1, keepdim=True)  # [B, 1, 3]
    x = x - cx
    y = y - cy
    w = torch.bmm(x.transpose(-1,-2), y)  # [B, 3, 3]
    if cpu:
        u, s, vh = torch.svd(w.cpu())
        u = u.to(x.device)
        s = s.to(x.device)
        vh = vh.to(x.device)
    else:
        u, s, vh = torch.svd(w)  # [B, 3, 3], [B, 3, 3], [B, 3, 3]
    ide = torch.eye(3).to(x.device).unsqueeze(0).repeat(y.shape[0], 1,1)
    ide[:,2,2] = torch.det(torch.bmm(vh, u.transpose(-1,-2)))
    R = torch.bmm(torch.bmm(vh, ide), u.transpose(-1,-2))
    t = cy -  torch.bmm(cx, R.transpose(-1,-2))  #[B, 1, 3]
    return R, t.transpose(-1,-2)

def ransac_rt(x, y, n=0, cpu=True):
    '''
    seems that n=0 is best. Cpu is faster for svd.
    x: [num, 3]
    y: [B, num, 3]
    :return R: [B, 3, 3]
    :return t: [B, 3, 1]
    '''
    num = y.shape[1]
    all_index = range(num)
    index = []
    if n == 0:
        R, t = solve_rot_and_trans(x,y, cpu)
        return R, t, None, None,None
    if n == 3:
        for i in range(num):
            for j in range(i+1, num):
                for k in range(j+1, num):
                    index.append([i,j,k])
    elif n == 4:
        for i in range(num):
            for j in range(i + 1, num):
                for k in range(j + 1, num):
                    for l in range(k + 1, num):
                        index.append([i, j, k, l])
    else:
        raise NotImplementedError
    R_lst = []
    t_lst = []
    error_lst = []
    for i in index:
        R, t = solve_rot_and_trans(x[:, i,:], y[:, i, :])
        out_index = list(set(list(all_index)) - set(list(i)))
        error_lst.append((y[:, out_index, :] - torch.bmm(x[:, out_index,:], R.transpose(-1,-2))-t.transpose(-1,-2)).norm(dim=-1).mean())
        R_lst.append(R)
        t_lst.append(t)
    min_ind = np.argmin(error_lst)
    minR = R_lst[min_ind]
    minT = t_lst[min_ind]
    retR = torch.stack(R_lst, dim=1)
    retT = torch.stack(t_lst, dim=1)
    return minR, minT, retR, retT, error_lst

def handkp2palmkp(kp):
    '''
    :param kp: [B, kp_num, 3]
    :return ret: [B, 6, 3]
    '''
    if kp.shape[1] == 21:
        ind_lst = [0, 1,5,9,13,17]
        ret = kp[:,ind_lst,:]
    elif kp.shape[1] == 29:
        ind_lst = [0, 1, 5,6,7, 11,12,13, 17,18,19, 23,24,25]
        ret = kp[:,ind_lst,:]
    else:
        raise NotImplementedError
    return ret
