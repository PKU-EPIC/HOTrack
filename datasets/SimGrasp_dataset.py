import os
import sys
import numpy as np
from os.path import join as pjoin
import argparse
import torch
from third_party.mano.our_mano import OurManoLayer
from network.models.hand_utils import handkp2palmkp

base_dir = os.path.dirname(__file__)
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, '..'))

from configs.config import get_config
from data_utils import farthest_point_sample, mat_from_rvec, split_dataset, jitter_hand_mano, jitter_obj_pose, pose_list_to_dict, jitter_hand_kp, OBB

"""
ShapeNet data organization:
    img/
        ...
            depth.png
            mask.png
    preproc/
        xxx.npz
        ...
    objs/
    masks/
    SDF/
    splits
"""

category2scale = {
    'bottle_sim': 0.25,
    'bowl_sim': 0.25,
    'car_sim': 0.3,
}

def generate_shapenet_data(path, category, num_parts, num_points, obj_perturb_cfg, hand_jitter_config, device, handframe, mano_layer_right, load_pred_obj_pose=False,pred_obj_pose_dir=None):
    #read file
    cloud_dict = np.load(path, allow_pickle=True)["all_dict"].item()

    cam = cloud_dict['points']
    label = cloud_dict['labels']
    if len(cam) == 0:
        return None

    #shuffle
    n = cam.shape[0]
    perm = np.random.permutation(n)
    cam = cam[perm]
    label = label[perm]

    #filter
    hand_id = num_parts
    hand_idx = np.where(label == hand_id)[0]
    if len(hand_idx) == 0:
        return None
    else:
        hand_pcd = cam[hand_idx]
        sample_idx = farthest_point_sample(hand_pcd, num_points, device)
        hand_pcd = hand_pcd[sample_idx]

    obj_idx = np.where(label != hand_id)[0]
    if len(obj_idx) == 0:
        return None 
    else:
        obj_pcd = cam[obj_idx]
        sample_idx = farthest_point_sample(obj_pcd, num_points, device)
        obj_pcd = obj_pcd[sample_idx]

    # generate obj canonical point clouds
    obj_pose = cloud_dict['obj_pose']
    if num_parts == 1:
        obj_pose = [obj_pose]

    for i in range(num_parts):
        obj_pose[i]['translation'] = np.expand_dims(np.array(obj_pose[i]['translation']), axis=1)

    #generate hand canonical point clouds
    mano_pose = np.array(cloud_dict['hand_pose']['mano_pose'])
    hand_global_rotation = mat_from_rvec(mano_pose[:3])
    mano_trans = np.array(cloud_dict['hand_pose']['mano_trans'])
    mano_beta = np.array(cloud_dict['hand_pose']['mano_beta'])

    hand_template, hand_kp = mano_layer_right.forward(th_pose_coeffs=torch.FloatTensor(np.array(mano_pose).reshape(1, -1)).cuda(),
                                            th_trans=torch.FloatTensor(mano_trans).reshape(1, -1).cuda(),
                                            th_betas=torch.FloatTensor(mano_beta).reshape(1, -1).cuda(), original_version=True)
    beta = mano_beta.reshape(1,-1)
    rest_pose = np.zeros_like(mano_pose)
    hand_kp = hand_kp[0].cpu()
    world_trans = hand_kp[0] 
    hand_kp = hand_kp.numpy()

    _, template_kp = mano_layer_right.forward(th_pose_coeffs=(torch.zeros((1, 48))).cuda(), th_trans=(torch.zeros((1, 3))).cuda(),
                                            th_betas=torch.FloatTensor(mano_beta).reshape(1, -1).cuda())
    palm_template = handkp2palmkp(template_kp)
    palm_template = palm_template[0].float().cpu()

    jittered_hand_kp = jitter_hand_kp(hand_kp, hand_jitter_config)

    pose_perturb_cfg = {'type': obj_perturb_cfg['type'],
                        'scale': obj_perturb_cfg['s'],
                        'translation': obj_perturb_cfg['t'],  # Now pass the sigma of the norm
                        'rotation': np.deg2rad(obj_perturb_cfg['r'])}
    jittered_obj_pose_lst = []
    for i in range(num_parts):
        jittered_obj_pose = jitter_obj_pose(obj_pose[i], pose_perturb_cfg)
        jittered_obj_pose_lst.append(jittered_obj_pose)

    full_data = {
        'hand_points': hand_pcd,
        'obj_points': obj_pcd,
        'jittered_obj_pose': pose_list_to_dict(jittered_obj_pose_lst),     # list
        'gt_obj_pose': pose_list_to_dict(obj_pose),                        # list
        'jittered_hand_kp': jittered_hand_kp,
        'gt_hand_kp': hand_kp,
        'gt_hand_pose':{'translation':np.expand_dims(world_trans, axis=1),
                          'scale': 0.2,
                          'rotation': np.array(hand_global_rotation),
                          'mano_pose':mano_pose,
                          'mano_trans':mano_trans,
                          'palm_template': palm_template,
                          'mano_beta': beta[0],
                          },
        'category': category,
        'file_name':cloud_dict['file_name'],
        'projection': { 'cx': 512/2, 'cy': 424/2, 'fx': -1.4343544 * 512/ 2.0, 'fy': 1.7320507 * 424 / 2.0, 'h': 424, 'w': 512}   # don't need for hand tracking
    }
    full_data['gt_obj_pose']['up_and_down_sym'] = False
    if load_pred_obj_pose:
        pred_obj_result_pth = os.path.join(pred_obj_pose_dir, '%s_%s.pkl'%(category, path.split('/')[-1][:-8]))
        tmp = np.load(pred_obj_result_pth, allow_pickle=True)
        pred_dict = tmp
        del tmp
        pred_obj_pose_lst = pred_dict['pred_obj_poses']
        frame_id = int(path.split('/')[-1][-7:-4])
        pred_obj_pose = {
            'rotation': pred_obj_pose_lst[frame_id]['rotation'].squeeze(),
            'translation': pred_obj_pose_lst[frame_id]['translation'].squeeze(),
        }
        full_data['pred_obj_pose'] = pred_obj_pose
    if handframe == 'OBB':
        _,full_data['OBB_pose'] = OBB(cam_points) 
        if full_data['OBB_pose']['scale'] < 0.001:
            return None 
   
    return full_data

class SimGraspDataset:
    def __init__(self, cfg, mode):
        self.cfg = cfg
        self.root_dset = cfg['data_cfg']['basepath']
        self.obj_cat_lst = cfg['obj_category']
        self.load_pred_obj_pose = cfg['use_pred_obj_pose']
        self.handframe = cfg['network']['handframe']
        if 'pred_obj_pose_dir' in cfg:
            self.pred_obj_pose_dir = cfg['pred_obj_pose_dir']
        else:
            self.pred_obj_pose_dir = None

        self.mano_layer_right = OurManoLayer(mano_root=cfg['mano_root'], side='right').cuda()
        self.file_list = []
        self.num_parts = {}
        for cat in self.obj_cat_lst:
            self.num_parts[cat] = self.cfg['data_cfg'][cat]['num_parts']
            read_folder = pjoin(self.root_dset, 'preproc', cat, 'seq')
            splits_folder = pjoin(self.root_dset, "splits", cat, 'seq')
            use_txt = pjoin(splits_folder, f"{mode}.txt")
            splits_ready = os.path.exists(use_txt)
            if not splits_ready:
                if 'train_val_split' in self.cfg['data_cfg'][cat]:
                    split = self.cfg['data_cfg'][cat]['train_val_split']
                    train_ins_lst = ['%05d' % i for i in range(split[0])]
                    test_ins_lst = ['%05d' % i for i in range(split[0], split[0] + split[1])]
                else:
                    train_ins_lst = None
                    test_ins_lst = self.cfg['data_cfg'][cat]['test_list']
                split_dataset(splits_folder, read_folder, test_ins_lst, train_ins_lst)
            with open(use_txt, "r", errors='replace') as fp:
                lines = fp.readlines()
                file_list = [pjoin(read_folder, i.strip()) for i in lines]
            self.file_list.extend(file_list)

        self.len = len(self.file_list)
        print(f"mode: {mode}, data number: {self.len}, obj_lst: {self.obj_cat_lst}")

    def __getitem__(self, index):
        path = self.file_list[index]
        ins = self.file_list[index].split('/')[-1].split('_')[0]
        category = self.file_list[index].split('/')[-3]
        num_parts = self.num_parts[category]
        
        full_data = generate_shapenet_data(
                                            path, 
                                            category, 
                                            num_parts, 
                                            self.cfg['num_points'], 
                                            self.cfg['obj_jitter_cfg'],
                                            self.cfg['hand_jitter_cfg'],
                                            self.cfg['device'], 
                                            self.handframe,
                                            mano_layer_right=self.mano_layer_right, 
                                            load_pred_obj_pose=self.load_pred_obj_pose,
                                            pred_obj_pose_dir=self.pred_obj_pose_dir,
                                            )
        
        return full_data

    def __len__(self):
        return self.len


def visualize_data(data_dict, category):
    from vis_utils import plot3d_pts

    # import torch
    # full_obj = data_dict['full_obj']

    # mano_pose = data_dict['gt_hand_pose']['mano_pose']
    # mano_trans = data_dict['gt_hand_pose']['mano_trans']
    # mano_layer_right = ManoLayer(
    #         mano_root=mano_root , side='right', use_pca=False, ncomps=45, flat_hand_mean=True)
    # hand_vertices, _ = mano_layer_right.forward(th_pose_coeffs=torch.FloatTensor(np.array(mano_pose).reshape(1, -1)),
    #                                             th_trans=torch.FloatTensor(mano_trans).reshape(1, -1))
    # hand_vertices = hand_vertices.cpu().data.numpy()[0] / 1000
    hand_vertices = data_dict['gt_hand_pose']['hand_template']
    hand_vertices = np.matmul(hand_vertices - data_dict['gt_hand_pose']['translation'][:,0], data_dict['gt_hand_pose']['rotation'])*5
    visibility = data_dict['gt_hand_pose']['visibility']
    visible_template = hand_vertices[visibility]
    invisible_template = hand_vertices[~visibility]

    hand_pcd = data_dict['hand_points']
    obj_pcd = data_dict['obj_points']
    hand_pcd = np.matmul(hand_pcd - data_dict['gt_hand_pose']['translation'][:,0], data_dict['gt_hand_pose']['rotation'])*5
    obj_pcd = np.matmul(obj_pcd - data_dict['gt_hand_pose']['translation'][:,0], data_dict['gt_hand_pose']['rotation'])*5

    # full_obj = np.matmul(full_obj - data_dict['gt_hand_pose']['translation'][:,0], data_dict['gt_hand_pose']['rotation'])*5
    print(data_dict['file_name'])

    plot3d_pts([[hand_vertices, obj_pcd], [hand_pcd, obj_pcd], [visible_template], [invisible_template]],
               show_fig=False, save_fig=True,
               save_folder=pjoin('shapenet_data_vis', category),
               save_name=data_dict['file_name'])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='2.28_hand_joint.yml', help='path to config.yml')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    cfg = get_config(args, save=False)
    dataset = SimGraspDataset(cfg, args.mode)
    print(len(dataset))
    mano_diff_lst = []
    trans_diff_lst = []

    for i in range(5):
        data_dict = dataset[i * 2000]
        visualize_data(data_dict, 'bottle')
