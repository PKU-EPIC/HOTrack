import os
import sys
import numpy as np
from os.path import join as pjoin
import argparse
import yaml
from PIL import Image
import open3d as o3d
base_dir = os.path.dirname(__file__)
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, '..'))
sys.path.append(os.path.join(base_dir, '..', '..'))
from network.models.hand_utils import handkp2palmkp
from configs.config import get_config
from data_utils import farthest_point_sample, jitter_obj_pose, mat_from_rvec, pose_list_to_dict,jitter_hand_kp,OBB, matrix_to_unit_quaternion
import torch
from manopth.manolayer import ManoLayer
from copy import deepcopy

"""
DexYCB data organization:
    20200709-subject-01
    ...
    20201022-subject-10
    calibration
    SDF
    splits
"""


invalid_lst = ['20200820-subject-03+20200820_143206+839512060362',
    '20200820-subject-03+20200820_143206+840412060917',
    '20200820-subject-03+20200820_143206+932122061900',
    '20201002-subject-08+20201002_111616+841412060263',
    '20201002-subject-08+20201002_111616+839512060362',
    '20201002-subject-08+20201002_111616+840412060917',
    '20201022-subject-10+20201022_113502+839512060362',
    '20200820-subject-03+20200820_141302+841412060263',
    '20200820-subject-03+20200820_141302+840412060917',
    '20200908-subject-05+20200908_143832+839512060362',
    '20200908-subject-05+20200908_143832+932122060857',
    '20200908-subject-05+20200908_145430+932122062010',
    '20200928-subject-07+20200928_145424+836212060125',
    '20201002-subject-08+20201002_110425+841412060263',
    '20201015-subject-09+20201015_143338+841412060263',
    '20201015-subject-09+20201015_144651+841412060263',
    '20201015-subject-09+20201015_143338+932122062010',
    '20201015-subject-09+20201015_143338+932122060861',
    '20201015-subject-09+20201015_143338+839512060362',
    '20200928-subject-07+20200928_145204+836212060125']

_YCB_CLASSES = {
     1: '002_master_chef_can',
     2: '003_cracker_box',
     3: '004_sugar_box',
     4: '005_tomato_soup_can',
     5: '006_mustard_bottle',
     6: '007_tuna_fish_can',
     7: '008_pudding_box',
     8: '009_gelatin_box',
     9: '010_potted_meat_can',
    10: '011_banana',
    11: '019_pitcher_base',
    12: '021_bleach_cleanser',
    13: '024_bowl',
    14: '025_mug',
    15: '035_power_drill',
    16: '036_wood_block',
    17: '037_scissors',
    18: '040_large_marker',
    19: '051_large_clamp',
    20: '052_extra_large_clamp',
    21: '061_foam_brick',
}

def load_point_clouds_DexYCB(cam_in_path, dpt_pth, labels, obj_id, obj_trans, hand_center, obj_scale):
    fs = open(cam_in_path, encoding="UTF-8")
    intri_anno = yaml.load(fs, Loader=yaml.FullLoader)
    tmp = intri_anno['color']
    color_cam_intrinsics = intri_anno['color']
    K = np.eye(3)
    K[0][0] = tmp['fx']
    K[1][1] = tmp['fy']
    K[0][2] = tmp['ppx']
    K[1][2] = tmp['ppy']
    
    with Image.open(dpt_pth) as di:
        dpt_map = np.array(di)/1000

    hand_mask = (labels == 255)
    obj_mask = (labels == obj_id)

    obj_depth = np.array(dpt_map * obj_mask).astype(np.float32)
    hand_depth = np.array(dpt_map * hand_mask).astype(np.float32)

    depth3d_obj = o3d.geometry.Image(obj_depth)
    intrinsics = o3d.camera.PinholeCameraIntrinsic(640, 480, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    obj_pcd = o3d.geometry.PointCloud.create_from_depth_image(depth3d_obj, intrinsics, stride=2)
    obj_pcd = np.asarray(obj_pcd.points)

    norm = np.linalg.norm(obj_pcd - obj_trans.reshape(1, 3), axis=-1)
    obj_pcd = obj_pcd[norm < (obj_scale / 2)]
    # obj_pcd = obj_pcd[norm < 0.25]

    depth3d_hand = o3d.geometry.Image(hand_depth)
    hand_pcd = o3d.geometry.PointCloud.create_from_depth_image(depth3d_hand, intrinsics, stride=2)
    hand_pcd = np.asarray(hand_pcd.points)
    norm = np.linalg.norm(hand_pcd - hand_center.reshape(1, 3), axis=-1)
    hand_pcd = hand_pcd[norm < 0.15]

    return obj_pcd, hand_pcd, color_cam_intrinsics

def generate_dexycb_data(root_dir, seq, id, num_points, device, mano_layer_right, obj_perturb_cfg, pred_obj_pose_dir, start_frame, load_pred_obj_pose, hand_jitter_config, handframe):
    serial = seq.split('/')[-1]
    cam_in_path = pjoin(root_dir, 'calibration/intrinsics/%s_640x480.yml' % serial)
    dpt_pth = pjoin(root_dir, '%s/aligned_depth_to_color_%06d.png' % (seq, id))
    anno_pth = pjoin(root_dir, '%s/labels_%06d.npz' % (seq, id))
    
    anno = np.load(anno_pth)
    labels = anno['seg']
    subject, scene = seq.split('/')[0], seq.split('/')[1]
    obj_info_path = pjoin(root_dir, '%s/%s/meta.yml' % (subject, scene))
    f = open(obj_info_path, 'r')
    cfg = yaml.load(f, Loader=yaml.FullLoader)
    index_in_scene = cfg['ycb_grasp_ind']
    obj_id = cfg['ycb_ids'][index_in_scene]
    
    # obj pose
    obj_trans = anno['pose_y'][index_in_scene][:, 3]  # [3]
    obj_rot = anno['pose_y'][index_in_scene][:, :3]  # [3,3]
    obj_name = _YCB_CLASSES[obj_id]
    scale_pth = pjoin(root_dir, '../YCB/SDF/NormalizationParameters', obj_name, 'textured_simple.npz')
    obj_scale = 2 / np.load(scale_pth)['scale']

    # hand pose
    mano_pose = anno['pose_m'][0][:48]
    rotvec = mano_pose[:3]
    hand_global_rotation = mat_from_rvec(rotvec)
    mano_trans = anno['pose_m'][0][48:51]
    mano_calib_file = os.path.join(pjoin(root_dir, "calibration", f"mano_{cfg['mano_calib'][0]}", "mano.yml"))
    with open(mano_calib_file, 'r') as f:
        mano_calib = yaml.load(f, Loader=yaml.FullLoader)
    mano_beta = mano_calib['betas']

    handv, hand_kp = mano_layer_right.forward(th_betas=torch.FloatTensor(mano_beta).reshape(1, -1),
                                         th_pose_coeffs=torch.FloatTensor(np.array(mano_pose).reshape(1, -1)),
                                         th_trans=torch.FloatTensor(mano_trans).reshape(1, -1)
                                         )
    hand_kp = hand_kp[0].cpu().float().numpy() / 1000
    handv = handv[0].cpu().float().numpy() / 1000
    zeroR_pose = deepcopy(mano_pose)
    zeroR_pose[:3] = 0
    _, template_kp = mano_layer_right.forward(th_betas=torch.FloatTensor(mano_beta).reshape(1, -1),
                                         th_pose_coeffs=torch.FloatTensor(np.array(zeroR_pose).reshape(1, -1)),
                                         th_trans=torch.zeros((1, 3)))
    template_kp = (template_kp / 1000).cpu().float().numpy()
    template_translation = template_kp[0,0:1]
    canonical_template_kp = template_kp - template_translation

    palm_template = handkp2palmkp(canonical_template_kp)[0]
    hand_global_trans = hand_kp[0]
    hand_center = hand_kp[9]

    obj_pcld, hand_pcld, color_cam_intrinsics = load_point_clouds_DexYCB(cam_in_path, dpt_pth, labels, obj_id, obj_trans, hand_center, obj_scale)

    # sampling
    if len(hand_pcld) == 0:
        return None  
    sample_idx = farthest_point_sample(hand_pcld, num_points, device)
    hand_pcld = hand_pcld[sample_idx]

    if len(obj_pcld) == 0:
        return None  
    sample_idx = farthest_point_sample(obj_pcld, num_points, device)
    obj_pcld = obj_pcld[sample_idx]

    obj_pose = {
        'translation': obj_trans[:][:,None],
        'rotation': obj_rot,
        'scale': obj_scale,
    }

    pose_perturb_cfg = {'type': obj_perturb_cfg['type'],
                        'scale': obj_perturb_cfg['s'],
                        'translation': obj_perturb_cfg['t'],  # Now pass the sigma of the norm
                        'rotation': np.deg2rad(obj_perturb_cfg['r'])}
    jittered_obj_pose_lst = []
    jittered_obj_pose = jitter_obj_pose(obj_pose, pose_perturb_cfg)
    jittered_obj_pose_lst.append(jittered_obj_pose)

    jittered_hand_kp = jitter_hand_kp(hand_kp, hand_jitter_config)
    full_data = {
        'hand_points': hand_pcld,
        'obj_points': obj_pcld,
        'gt_obj_pose': pose_list_to_dict([obj_pose]),
        'jittered_obj_pose':pose_list_to_dict(jittered_obj_pose_lst), # buggy!
        'category': obj_name,
        'gt_hand_pose':{
            'mano_trans': np.array(mano_trans).reshape(3), # 3
            'scale': 0.2,  
            'rotation': np.array(hand_global_rotation).reshape(1, 3, 3),
            'mano_pose':np.array(mano_pose),
            'translation': np.array(hand_global_trans),
            'mano_beta': np.array(mano_beta),
            'palm_template': np.array(palm_template)
        },
        'file_name': ('%s/%06d' % (seq, id)).replace('/','+'),
        'jittered_hand_kp': jittered_hand_kp,
        'gt_hand_kp': hand_kp,
        'projection': {
            'fx': color_cam_intrinsics['fx'], 
            'fy': color_cam_intrinsics['fy'], 
            'cx': color_cam_intrinsics['ppx'], 
            'cy': color_cam_intrinsics['ppy'], 
            'w': 640, 
            'h': 480},
    }
    if handframe == 'OBB':
        _,full_data['OBB_pose'] = OBB(hand_pcld) 
        if full_data['OBB_pose']['scale'] < 0.001:
            return None 

    if 'can' in full_data['category'] or 'box' in full_data['category']:
        full_data['gt_obj_pose']['up_and_down_sym'] = True
    else:
        full_data['gt_obj_pose']['up_and_down_sym'] = False
    
    if load_pred_obj_pose:
        pred_obj_result_pth = os.path.join(pred_obj_pose_dir, '%s+%06d.pkl' % (seq.replace('/', '+'), start_frame))
        pred_dict = np.load(pred_obj_result_pth, allow_pickle=True)
        pred_obj_pose_lst = pred_dict['pred_obj_poses']
        frame_id = id - start_frame
        pred_obj_pose = {
            'rotation': pred_obj_pose_lst[frame_id]['rotation'].squeeze(),
            'translation': pred_obj_pose_lst[frame_id]['translation'].squeeze(),
        }
        full_data['pred_obj_pose'] = pred_obj_pose
    
    return full_data

class DexYCBDataset:
    def __init__(self, cfg, mode):
        self.cfg = cfg
        self.mode = mode
        self.category_lst = cfg['obj_category']
        self.root_dir = cfg['data_cfg']['basepath']
        if 'pred_obj_pose_dir' in cfg:
            self.pred_obj_pose_dir = cfg['pred_obj_pose_dir']
        else:
            self.pred_obj_pose_dir = None
        self.device = cfg['device']
        self.num_points = cfg['num_points']
        self.hand_jitter_config = cfg['hand_jitter_cfg']
        self.handframe = cfg['network']['handframe']
        print('----------------DexYCB dataset----------------')
        print('category: ', self.category_lst)
        print('mode: ', self.mode)
        print('Predicted object pose', self.pred_obj_pose_dir)

        self.seq_name_lst = []
        self.id_lst = []
        self.seq_start = []
        self.start_frame_lst = []
        self.load_pred_obj_pose = cfg['use_pred_obj_pose']

        cnt = 0
        for category in self.category_lst:
            our_clean_split = np.load(pjoin(self.root_dir, 'splits/%s_%s.npy' % (self.mode, category)),allow_pickle=True).item()
            for filename in our_clean_split.keys():
                if filename in invalid_lst:
                    continue
                self.seq_start.append(cnt)
                start_frame = int(our_clean_split[filename][0].split('.')[0])
                for frame in our_clean_split[filename]:
                    seq_name = filename.replace('+', '/')
                    self.seq_name_lst.append(seq_name)
                    self.id_lst.append(int(frame.split('.')[0]))
                    self.start_frame_lst.append(start_frame)
                    cnt += 1
        self.seq_start.append(cnt)
        self.len = len(self.id_lst)
        print('Sequence: ', len(self.seq_start) - 1)
        print('Len: ',self.len)
        self.manolayer = ManoLayer(mano_root=cfg['mano_root'], side='right',
                                   use_pca=True, ncomps=45, flat_hand_mean=False)

    def __getitem__(self, index):
        try:
            seq = self.seq_name_lst[index]
            id = self.id_lst[index]
            start_frame = self.start_frame_lst[index]
            full_data = generate_dexycb_data(self.root_dir, seq, id, self.num_points, self.device, 
                self.manolayer, self.cfg['obj_jitter_cfg'], self.pred_obj_pose_dir,
                start_frame, self.load_pred_obj_pose,self.hand_jitter_config, self.handframe)
            return full_data
        except:
            return self.__getitem__((index+1)%self.len)

    def __len__(self):
        return self.len


def visualize_data(data_dict, category):
    from vis_utils import plot3d_pts
    mano_pose = data_dict['gt_hand_pose']['mano_pose']
    trans = data_dict['gt_hand_pose']['mano_trans']
    beta = data_dict['gt_hand_pose']['mano_beta']
    mano_layer_right = ManoLayer(mano_root='/home/hewang/jiayi/manopth/mano/models', side='right',
                                   use_pca=True, ncomps=45, flat_hand_mean=False)
    hand_vertices, _ = mano_layer_right.forward(th_pose_coeffs=torch.FloatTensor(np.array(mano_pose).reshape(1, -1)),
                                                th_trans=torch.FloatTensor(trans).float().reshape(1, -1), 
                                                th_betas=torch.from_numpy(beta).float().unsqueeze(0))
    hand_vertices = hand_vertices.cpu().data.numpy()[0] / 1000
    hand_vertices = np.matmul(hand_vertices - data_dict['gt_hand_pose']['translation'].reshape(1, 3), data_dict['gt_hand_pose']['rotation'].reshape(3,3)) * 5
    hand_input = np.matmul(data_dict['pred_seg_hand_points'] - data_dict['gt_hand_pose']['translation'].reshape(1, 3), data_dict['gt_hand_pose']['rotation'].reshape(3,3)) * 5
    obj_input = np.matmul(data_dict['pred_seg_obj_points'] - data_dict['gt_hand_pose']['translation'].reshape(1, 3), data_dict['gt_hand_pose']['rotation'].reshape(3,3)) * 5
    print(data_dict['file_name'])

    plot3d_pts([[hand_vertices], [hand_input], [obj_input], [hand_input, obj_input]],
               show_fig=False, save_fig=True,
               save_folder='./DexYCB/',
               save_name=data_dict['file_name'])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='3.6_final_obj_box_test_DexYCB.yml', help='path to config.yml')
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--kind', type=str, default='seq', choices=['single_frame', 'seq'])
    return parser.parse_args()

if __name__ == '__main__':
    from torch.utils.data import Dataset, DataLoader
    import tqdm
    args = parse_args()
    cfg = get_config(args, save=False)
    dataset = DexYCBDataset(cfg,mode=args.mode, kind=args.kind)

    translation_lst = []
    rot_lst = []
    for i in range(400):
        full_data = dataset[i]
        translation = full_data['gt_hand_pose']['translation'] * 1000
        translation[-1] *= -1
        translation_lst.append(translation)

        rotation = full_data['gt_hand_pose']['rotation']
        quat = matrix_to_unit_quaternion(rotation).numpy()
        rot_lst.append(quat)

    translation_array = np.array(translation_lst)
    rot_array = np.array(rot_lst)
    np.savetxt('translation.txt', translation_array)
    np.savetxt('rot.txt', rot_array)
    exit(0)

    for i in range(10):
        visualize_data(dataset[i * 100], cfg['obj_category'])
    # torch.multiprocessing.set_start_method('spawn')
    # test_dataloader = DataLoader(dataset, batch_size=32, num_workers=8)
    # for i, data in tqdm.tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
    #     new = i
