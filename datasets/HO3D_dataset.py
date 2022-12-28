import os
import sys
import numpy as np
from os.path import join as pjoin
import torch
from third_party.mano.our_mano import OurManoLayer
from network.models.hand_utils import handkp2palmkp
base_dir = os.path.dirname(__file__)
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, '..'))

from data_utils import farthest_point_sample, mat_from_rvec, jitter_hand_kp, jitter_obj_pose, pose_list_to_dict
import cv2


"""
HO3D data organization:
    calibration  
    evaluation   
    evaluation.txt           
    manual_annotations  
    train
        ABF10
            depth  
            meta  
            rgb  
            seg
        ...
    train.txt
    splits
    SDF
"""

height, width = 480, 640
xmap = np.array([[j for i in range(width)] for j in range(height)])
ymap = np.array([[i for i in range(width)] for j in range(height)])

def read_depth_img(depth_filename):
    """Read the depth image in dataset and decode it"""

    depth_scale = 0.00012498664727900177
    depth_img = cv2.imread(depth_filename)
    dpt = depth_img[:, :, 2] + depth_img[:, :, 1] * 256
    dpt = dpt * depth_scale
    return dpt

def get_intrinsics(filename):
    with open(filename, 'r') as f:
        line = f.readline()
    line = line.strip()
    items = line.split(',')
    for item in items:
        if 'fx' in item:
            fx = float(item.split(':')[1].strip())
        elif 'fy' in item:
            fy = float(item.split(':')[1].strip())
        elif 'ppx' in item:
            ppx = float(item.split(':')[1].strip())
        elif 'ppy' in item:
            ppy = float(item.split(':')[1].strip())

    camMat = np.array([[fx, 0, ppx], [0, fy, ppy], [0, 0, 1]])
    return camMat


def dpt_2_cld(dpt, K):
    if len(dpt.shape) > 2:
        dpt = dpt[:, :, 0]
    msk_dp = dpt > 1e-6
    choose = msk_dp.flatten().nonzero()[0].astype(np.uint32)

    if len(choose) < 1:
        return None, None

    dpt_mskd = dpt.flatten()[choose][:, np.newaxis].astype(np.float32)
    xmap_mskd = xmap.flatten()[choose][:, np.newaxis].astype(np.float32)
    ymap_mskd = ymap.flatten()[choose][:, np.newaxis].astype(np.float32)

    pt2 = dpt_mskd
    cam_cx, cam_cy = K[0][2], K[1][2]
    cam_fx, cam_fy = K[0][0], K[1][1]
    pt0 = (ymap_mskd - cam_cx) * pt2 / cam_fx
    pt1 = (xmap_mskd - cam_cy) * pt2 / cam_fy
    cld = np.concatenate((pt0, pt1, pt2), axis=1)  # position in camera reference

    return cld, choose

def load_point_clouds(root_dir, seq, fID):
    path_depth = os.path.join(root_dir, 'train/%s/depth/%s.png' % (seq, fID))
    depth_raw = read_depth_img(path_depth)
    anno = get_anno(root_dir, seq, fID)
    if seq[-2].isnumeric():
        calibDir = os.path.join(root_dir, 'calibration', seq[:-1], 'calibration')
        K = get_intrinsics(os.path.join(calibDir, 'cam_{}_intrinsics.txt'.format(seq[-1]))).tolist()
    else:
        K = anno['camMat']

    mask_pth = os.path.join(root_dir, 'train/%s/seg/%s.png' % (seq, fID))
    mask = cv2.imread(mask_pth)
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    mask = mask.reshape(-1, 3)

    cld, choose = dpt_2_cld(depth_raw, K)
    cld[:, 1] *= -1
    cld[:, 2] *= -1
    mask = mask[choose, :]
    
    hand_idx = (mask[..., 0] == 255)
    obj_idx = (mask[..., 1] == 255)
    hand_pcld = cld[hand_idx]
    obj_pcld = cld[obj_idx]
    return hand_pcld, obj_pcld, K, anno

def get_anno(root_dir, seq, fID):
    import pickle
    f_name = pjoin(root_dir, 'train/%s/meta/%s.pkl' % (seq, fID))
    with open(f_name, 'rb') as f:
        try:
            pickle_data = pickle.load(f, encoding='latin1')
        except:
            pickle_data = pickle.load(f)

    return pickle_data

def shuffle_pcld(pcld):
    n = pcld.shape[0]
    perm = np.random.permutation(n)
    pcld = pcld[perm]
    return pcld

def generate_HO3D_data(mano_layer_right, root_dir, seq, fID, num_points, obj_perturb_cfg, hand_jitter_config, device, load_pred_obj_pose, pred_obj_pose_dir, start_frame, cur_frame):
    # get intrinsics and point cloud and annotations
    hand_pcld, obj_pcld, cam_Mat, anno = load_point_clouds(root_dir, seq, fID)
    cam_cx, cam_cy = cam_Mat[0][2], cam_Mat[1][2]
    cam_fx, cam_fy = cam_Mat[0][0], cam_Mat[1][1]

    # get object pose
    scale_pth = pjoin(root_dir, '../YCB/SDF/NormalizationParameters', anno['objName'], 'textured_simple.npz')
    scale = 2 / np.load(scale_pth)['scale']
    origin_obj_pose = {
        'translation': anno['objTrans'],
        'ID': seq[:-1],
        'rotation': cv2.Rodrigues(anno['objRot'])[0],
        'scale': scale,
        'CAD_ID': anno['objName'],
    }
    obj_pose = {}
    obj_pose['translation'] = np.expand_dims(np.array(origin_obj_pose['translation']), axis=1)
    obj_pose['rotation'] = origin_obj_pose['rotation']
    obj_pose['scale'] = origin_obj_pose['scale']

    # get hand pose
    mano_pose = np.array(anno['handPose'])
    hand_global_rotation = mat_from_rvec(mano_pose[:3])
    mano_trans = np.array(anno['handTrans'])

    # get hand keypoints gt
    reorder = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
    hand_kp = anno['handJoints3D']
    hand_kp = hand_kp[reorder]
    world_trans = hand_kp[0]

    # remove outliers
    obj_dis = np.linalg.norm(obj_pcld - obj_pose['translation'].transpose(-1,-2), axis=-1)
    foreground = np.where(obj_dis < 0.25)
    obj_pcld = obj_pcld[foreground]

    hand_dis = np.linalg.norm(hand_pcld - hand_kp[9], axis=-1)
    foreground = np.where(hand_dis < 0.15)
    hand_pcld = hand_pcld[foreground]
    
    # point cloud downsample
    sample_idx = farthest_point_sample(hand_pcld, num_points, device)
    hand_pcld = hand_pcld[sample_idx]

    sample_idx = farthest_point_sample(obj_pcld, num_points, device)
    obj_pcld = obj_pcld[sample_idx]

    # shuffle
    obj_pcld = shuffle_pcld(obj_pcld)
    hand_pcld = shuffle_pcld(hand_pcld)
    
    # jitter hand pose
    jittered_hand_kp = jitter_hand_kp(hand_kp, hand_jitter_config)

    # get hand template
    rest_pose = torch.zeros((1, 48))
    rest_pose[0, 3:] = torch.FloatTensor(mano_pose[3:])
    _, template_kp = mano_layer_right.forward(th_pose_coeffs=rest_pose, th_trans=torch.zeros((1, 3)), th_betas=torch.FloatTensor(anno['handBeta']).reshape(1, 10))
    palm_template = handkp2palmkp(template_kp)
    palm_template = palm_template[0].cpu().float()

    # jitter obj pose
    pose_perturb_cfg = {'type': obj_perturb_cfg['type'],
                        'scale': obj_perturb_cfg['s'],
                        'translation': obj_perturb_cfg['t'],  # Now pass the sigma of the norm
                        'rotation': np.deg2rad(obj_perturb_cfg['r'])}
    jittered_obj_pose_lst = []
    jittered_obj_pose = jitter_obj_pose(obj_pose, pose_perturb_cfg)
    jittered_obj_pose_lst.append(jittered_obj_pose)
    
    full_data = {
        'hand_points': hand_pcld,
        'obj_points': obj_pcld,
        'jittered_obj_pose': pose_list_to_dict(jittered_obj_pose_lst),
        'gt_obj_pose': pose_list_to_dict([obj_pose]),
        'jittered_hand_kp': jittered_hand_kp,
        'gt_hand_kp': hand_kp,
        'gt_hand_pose':{
                        'translation':world_trans,
                        'scale': 0.2,
                        'rotation': np.array(hand_global_rotation),
                        'mano_pose':mano_pose,
                        'mano_trans':mano_trans,
                        'mano_beta':  anno['handBeta'],
                        'palm_template': palm_template
                        },
        'category': anno['objName'],
        'file_name': '%s/%s' % (seq, fID),
        'projection': {'w':640, 'h':480, 'fx':-cam_fx,'fy':cam_fy,'cx':cam_cx,'cy':cam_cy},     # don't need for hand tracking
    }

    if load_pred_obj_pose:
        pred_obj_result_pth = os.path.join(pred_obj_pose_dir, '%s_%04d.pkl' % (seq.replace('/', '_'), start_frame))
        tmp = np.load(pred_obj_result_pth, allow_pickle=True)
        pred_dict = tmp
        del tmp
        pred_obj_pose_lst = pred_dict['pred_obj_poses']
        frame_id = cur_frame - start_frame
        pred_obj_pose = {
            'rotation': pred_obj_pose_lst[frame_id]['rotation'].squeeze(),
            'translation': pred_obj_pose_lst[frame_id]['translation'].squeeze(),
        }
        full_data['pred_obj_pose'] = pred_obj_pose

    if 'can' in full_data['category'] or 'box' in full_data['category']:
        full_data['gt_obj_pose']['up_and_down_sym'] = True
    else:
        full_data['gt_obj_pose']['up_and_down_sym'] = False

    return full_data

class HO3DDataset:
    def __init__(self, cfg, mode):
        print('HO3DDataset!')
        self.cfg = cfg
        self.root_dset = cfg['data_cfg']['basepath']
        self.category_lst = cfg['obj_category']
        self.load_pred_obj_pose = cfg['use_pred_obj_pose']
        self.handframe = cfg['network']['handframe']
        if 'pred_obj_pose_dir' in cfg:
            self.pred_obj_pose_dir = cfg['pred_obj_pose_dir']
        else:
            self.pred_obj_pose_dir = None

        self.mode = mode 

        self.seq_lst = []
        self.fID_lst = []
        self.seq_start = []
        self.start_frame_lst = []
        self.mano_layer_right = OurManoLayer(mano_root=cfg['mano_root'], side='right')
        test_data_dict = {}

        # Load sequence information from the split file
        for category in self.category_lst:
            split_file_name = 'finalv2_test_%s.npy' % category
            split_file_pth = pjoin(self.root_dset, 'splits', split_file_name)
            tmp_dict = np.load(split_file_pth, allow_pickle=True).item()
            for key, value in tmp_dict.items():
                test_data_dict[key] = value
        
        for seq in test_data_dict.keys():
            for segment in test_data_dict[seq].keys():
                idx_lst = test_data_dict[seq][segment]
                self.seq_start.append(len(self.fID_lst))
                self.seq_lst.extend([seq] * len(idx_lst))
                self.fID_lst.extend(idx_lst)
                self.start_frame_lst.extend([idx_lst[0]] * len(idx_lst))
        self.seq_start.append(len(self.fID_lst))
        
        self.len = len(self.seq_lst)
        print('HO3D mode %s: %d frames' % (self.mode, self.len))
        
    def __getitem__(self, index):
        seq = self.seq_lst[index]
        fID = '%04d' % self.fID_lst[index]
        start_frame = self.start_frame_lst[index] if self.load_pred_obj_pose else None 
        cur_frame = self.fID_lst[index]

        full_data = generate_HO3D_data(self.mano_layer_right, self.root_dset, seq, fID,
                                            self.cfg['num_points'], self.cfg['obj_jitter_cfg'],
                                            self.cfg['hand_jitter_cfg'],
                                            self.cfg['device'], 
                                            load_pred_obj_pose=self.load_pred_obj_pose, 
                                            pred_obj_pose_dir=self.pred_obj_pose_dir,
                                            start_frame=start_frame,
                                            cur_frame=cur_frame)
        return full_data

    def __len__(self):
        return self.len

def visualize_data(data_dict, category):
    from vis_utils import plot3d_pts

    mano_pose = data_dict['gt_hand_pose']['mano_pose']
    trans = data_dict['gt_hand_pose']['translation']
    beta = data_dict['gt_hand_pose']['beta']
    mano_layer_right = OurManoLayer()
    hand_vertices, _ = mano_layer_right.forward(th_pose_coeffs=torch.FloatTensor(np.array(mano_pose).reshape(1, -1)),
                                                th_trans=torch.FloatTensor(trans).reshape(1, -1), th_betas=torch.from_numpy(beta).unsqueeze(0))
    hand_vertices = hand_vertices.cpu().data.numpy()[0]
    hand_vertices = np.matmul(hand_vertices - data_dict['gt_hand_pose']['translation'][:,0], data_dict['gt_hand_pose']['rotation']) * 5
    input_pc = np.matmul(data_dict['points'] - data_dict['gt_hand_pose']['translation'][:,0], data_dict['gt_hand_pose']['rotation'])*5
    print(data_dict['file_name'])

    plot3d_pts([[hand_vertices], [input_pc], [hand_vertices, input_pc]],
               show_fig=False, save_fig=True,
               save_folder=pjoin('HO3D', category),
               save_name=data_dict['file_name'])
