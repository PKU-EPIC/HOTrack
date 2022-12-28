import numpy as np
import torch
import os
import sys
from os.path import join as pjoin
BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, BASEPATH)
sys.path.insert(0, pjoin(BASEPATH, '..'))

from network.models.pointnet_utils import farthest_point_sample as farthest_point_sample_cuda
import transforms3d as tf
from copy import deepcopy 

def pose_list_to_dict(pose_lst):  # [{'scale': [1], 'translation': [3, 1], 'rotation': [3, 3]} * P]
    keys = list(pose_lst[0].keys())
    pose_dict = {key: np.stack([p[key] for p in pose_lst], axis=0) for key in keys}
    return pose_dict

def normalize(q):
    assert q.shape[-1] == 4
    norm = q.norm(dim=-1, keepdim=True)
    return q.div(norm)

def assert_normalized(q, atol=1e-3):
    assert q.shape[-1] == 4
    norm = q.norm(dim=-1)
    norm_check =  (norm - 1.0).abs()
    try:
        assert torch.max(norm_check) < atol
    except:
        print("normalization failure: {}.".format(torch.max(norm_check)))
        return -1
    return 0

def unit_quaternion_to_matrix(q):
    assert_normalized(q)
    w, x, y, z= torch.unbind(q, dim=-1)
    matrix = torch.stack(( 1 - 2*y*y - 2*z*z,  2*x*y - 2*z*w,      2*x*z + 2*y* w,
                        2*x*y + 2*z*w,      1 - 2*x*x - 2*z*z,  2*y*z - 2*x*w,
                        2*x*z - 2*y*w,      2*y*z + 2*x*w,      1 - 2*x*x -2*y*y),
                        dim=-1)
    matrix_shape = list(matrix.shape)[:-1]+[3,3]
    return matrix.view(matrix_shape).contiguous()

def matrix_to_unit_quaternion(matrix):
    assert matrix.shape[-1] == matrix.shape[-2] == 3
    if not isinstance(matrix, torch.Tensor):
        matrix = torch.tensor(matrix)

    trace = 1 + matrix[..., 0, 0] + matrix[..., 1, 1] + matrix[..., 2, 2]
    trace = torch.clamp(trace, min=0.)
    r = torch.sqrt(trace)
    s = 1.0 / (2 * r + 1e-7)
    w = 0.5 * r
    x = (matrix[..., 2, 1] - matrix[..., 1, 2])*s
    y = (matrix[..., 0, 2] - matrix[..., 2, 0])*s
    z = (matrix[..., 1, 0] - matrix[..., 0, 1])*s

    q = torch.stack((w, x, y, z), dim=-1)

    return normalize(q)

def noisy_rot_matrix(matrix, rad, type='normal'):
    if type == 'normal':
        theta = torch.abs(torch.randn_like(matrix[..., 0, 0])) * rad
    elif type == 'uniform':
        theta = torch.rand_like(matrix[..., 0, 0]) * rad
    quater = matrix_to_unit_quaternion(matrix)
    new_quater = jitter_quaternion(quater, theta.unsqueeze(-1))
    new_mat = unit_quaternion_to_matrix(new_quater)
    return new_mat

def jitter_quaternion(q, theta):  #[Bs, 4], [Bs, 1]
    new_q = generate_random_quaternion(q.shape).to(q.device)
    dot_product = torch.sum(q*new_q, dim=-1, keepdim=True)  #
    shape = (tuple(1 for _ in range(len(dot_product.shape) - 1)) + (4, ))
    q_orthogonal = normalize(new_q - q * dot_product.repeat(*shape))
    # theta = 2arccos(|p.dot(q)|)
    # |p.dot(q)| = cos(theta/2)
    tile_theta = theta.repeat(shape)
    jittered_q = q*torch.cos(tile_theta/2) + q_orthogonal*torch.sin(tile_theta/2)

    return jittered_q

def generate_random_quaternion(quaternion_shape):
    assert quaternion_shape[-1] == 4
    rand_norm = torch.randn(quaternion_shape)
    rand_q = normalize(rand_norm)
    return rand_q

def jitter_obj_pose(part, cfg):
    rand_type = cfg['type']  # 'uniform' or 'normal' --> we use 'normal'

    def random_tensor(base):
        if rand_type == 'uniform':
            return torch.rand_like(base) * 2.0 - 1.0
        elif rand_type == 'normal':
            return torch.randn_like(base)

    new_part = {}
    key = 'rotation'
    fl = 0
    if isinstance(part[key], np.ndarray):
        fl = 1
        part['rotation'] = torch.tensor(part['rotation'])
        part['translation'] = torch.tensor(part['translation'])
        part['scale'] = torch.tensor(part['scale'])
    new_part[key] = noisy_rot_matrix(part[key], cfg[key], type=rand_type).reshape(part[key].shape)
    key = 'scale'
    new_part[key] = part[key] + random_tensor(part[key]) * cfg[key]
    key = 'translation'
    norm = random_tensor(part['scale']) * cfg[key]  # [B, P]
    direction = random_tensor(part[key].squeeze(-1))  # [B, P, 3]
    direction = direction / torch.clamp(direction.norm(dim=-1, keepdim=True), min=1e-9)  # [B, P, 3] unit vecs
    new_part[key] = part[key] + (direction * norm.unsqueeze(-1)).unsqueeze(-1)  # [B, P, 3, 1]
    if fl:
        new_part['rotation'] = new_part['rotation'].numpy()
        new_part['translation'] = new_part['translation'].numpy()
        new_part['scale'] = new_part['scale'].numpy()
    return new_part

def mat_from_rvec(rvec):
    angle = np.linalg.norm(rvec)
    axis = np.array(rvec).reshape(3) / angle if angle != 0 else [0, 0, 1]
    mat = tf.axangles.axangle2mat(axis, angle)
    return np.matrix(mat)

def rvec_from_mat(mat):
    axis, angle = tf.axangles.mat2axangle(mat, unit_thresh=1e-03)
    rvec = axis * angle
    return rvec

def jitter_hand_kp(hand_kp, hand_jitter_config):
    rand_jitter_scale = hand_jitter_config['rand_scale']
    rand_type = hand_jitter_config['rand_type']
    kp_num = hand_kp.shape[-2]
    if rand_type == 'uniform':
        palm_noise = (torch.rand(size=(kp_num*3, )) * 2 - 1) * rand_jitter_scale
    elif rand_type == 'normal':
        palm_noise = torch.randn(size=(kp_num*3, )) * rand_jitter_scale
    else:
        raise NotImplementedError
    palm_noise = palm_noise.reshape(kp_num, 3)
    if isinstance(hand_kp, np.ndarray):
        jittered_hand_kp = hand_kp + palm_noise.numpy()
    else:
        jittered_hand_kp = hand_kp + palm_noise
    return jittered_hand_kp

def jitter_hand_mano(initial_rot_mat, initial_theta, initial_trans, initial_beta, hand_jitter_config):
    '''
            initial_theta: 45
            initial_rot_mat: 3*3
    '''
    noisy_rot_mat = noisy_rot_matrix(initial_rot_mat, hand_jitter_config['global_rotation'])
    noisy_axisangle = rvec_from_mat(noisy_rot_mat)

    jittered_trans = initial_trans + np.random.normal(scale=np.ones(3)*hand_jitter_config['global_translation'])
    jittered_beta = initial_beta + np.random.normal(scale=np.ones(10)*hand_jitter_config['beta'])

    scale = (np.ones(45).reshape(15, 3) * np.array([hand_jitter_config['x'], hand_jitter_config['y'], hand_jitter_config['z']])).reshape(45)
    mano_noise = np.random.normal(scale=scale)

    noisy_mano = np.zeros(48)
    noisy_mano[:3] = noisy_axisangle
    noisy_mano[3:] = initial_theta + mano_noise
    return noisy_mano, jittered_trans, jittered_beta

def OBB(x):
    '''
    transform x to the Oriented Bounding Box frame
    x: [N, 3]
    obb_x: [N, 3]
    '''
    x = deepcopy(x)
    #pca X
    n = x.shape[0]
    trans = x.mean(axis=0)
    x -= trans  
    C = np.dot(x.T, x) / (n - 1)
    # Eigen decomposition
    eigen_vals, eigen_vecs = np.linalg.eig(C)
    R = np.eye(3)
    max_ind = np.argmax(eigen_vals)        
    min_ind = np.argmin(eigen_vals)        
    R[:, 0] = eigen_vecs[:, max_ind]
    R[:, 2] = eigen_vecs[:, min_ind]

    R[:, 1] = np.cross(R[:, 2], R[:, 0])
    R[:, 1] = R[:, 1] / np.linalg.norm(R[:, 1])

    rotated_x = x @ R

    scale = 1.2     #???? don't know why...
    bbox_len = scale * (rotated_x.max(axis=0) - rotated_x.min(axis=0))
    #only use x-axis length
    normalized_x = rotated_x / bbox_len[0]

    T = normalized_x.mean(axis=0)
    obb_x = normalized_x - T
    record = {'rotation':R, 'translation': trans[:,None]+((R@T[:,None])* bbox_len[0]), 'scale': bbox_len[0]}  #R^(-1)(X-T)/s = obb_x
    return obb_x, record

def split_dataset(split_folder, read_folder, test_ins_lst, train_ins_lst=None):
    os.makedirs(split_folder, exist_ok=True)
    train_split_path = pjoin(split_folder, 'train.txt')
    test_split_path = pjoin(split_folder, 'test.txt')
    all_path = os.listdir(read_folder)
    all_path.sort()

    if train_ins_lst is None:
        train_split = [i for i in all_path if i.split('_')[0] not in test_ins_lst]
        test_split = [i for i in all_path if i.split('_')[0] in test_ins_lst]
    else:
        assert len(list(set([i.split('_')[0] for i in all_path]))) <= len(train_ins_lst) + len(test_ins_lst)
        train_split = [i for i in all_path if i.split('_')[0] in train_ins_lst]
        test_split = [i for i in all_path if i.split('_')[0] in test_ins_lst]

    with open(train_split_path, 'w') as f:
        f.write('\n'.join(train_split))
    with open(test_split_path, 'w') as f:
        f.write('\n'.join(test_split))

    return True

def farthest_point_sample(xyz, npoint, device):
    """
    Input:
        xyz: pointcloud data, [N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint]
    """
    if torch.cuda.is_available():
        if len(xyz) > 5 * npoint:
            idx = np.random.permutation(len(xyz))[:5 * npoint]
            torch_xyz = torch.tensor(xyz[idx]).float().to(device).reshape(1, -1, 3)
            torch_idx = farthest_point_sample_cuda(torch_xyz, npoint)
            torch_idx = torch_idx.cpu().numpy().reshape(-1)
            idx = idx[torch_idx]
        else:
            torch_xyz = torch.tensor(xyz).float().to(device).reshape(1, -1, 3)
            torch_idx = farthest_point_sample_cuda(torch_xyz, npoint)
            idx = torch_idx.reshape(-1).cpu().numpy()
        return idx
    else:
        print('FPS on CPU: use random sampling instead')
        idx = np.random.permutation(len(xyz))[:npoint]
        return idx

    N, C = xyz.shape
    centroids = np.zeros((npoint, ), dtype=int)
    distance = np.ones((N, )) * 1e10
    farthest = np.random.randint(0, N, dtype=int)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :].reshape(1, C)
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    return centroids



