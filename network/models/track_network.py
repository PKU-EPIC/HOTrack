import sys
import os
from os.path import join as pjoin

from matplotlib.pyplot import flag
BASEPATH = os.path.dirname(__file__)
sys.path.insert(0, BASEPATH)
sys.path.insert(0, pjoin(BASEPATH, '..'))
sys.path.insert(0, pjoin(BASEPATH, '..', '..'))

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from hand_network import *
from utils import add_dict, merge_dict, ensure_dirs, cvt_numpy
import pickle
from copy import deepcopy
import trimesh
from optimization_obj import InsCS2CatCS, gf_optimize_obj, get_RT
from optimization_hand import gf_optimize_hand_pose, gf_optimize_hand_shape
from third_party.mano.our_mano import OurManoLayer
from datasets.data_utils import farthest_point_sample
from pose_utils.part_dof_utils import eval_part_full
name_category_lst = {
    '002_master_chef_can': 'can',
    '005_tomato_soup_can': 'can',
    '010_potted_meat_can': 'can',
    '003_cracker_box': 'box',
    '004_sugar_box': 'box',
    '008_pudding_box': 'box',
    '009_gelatin_box': 'box',
    '024_bowl': 'bowl',
    '006_mustard_bottle': 'bottle',
    '021_bleach_cleanser': 'bottle',
}


def load_obj_for_opt(root_dir, dataset_name, sdf_code_source, seq_frame, instance):
    if dataset_name == 'HO3D':
        saved_model_pth = pjoin(root_dir, '../SimGrasp/SDF/examples/bottle_sim/ModelParameters/2000.pth')      
        normalization_pth = os.path.join(root_dir, '../YCB/SDF/NormalizationParameters/%s/textured_simple.npz' % instance)
        normalization_params = np.load(normalization_pth)
        gt_mesh_path = os.path.join(root_dir, f'../YCB/models/{instance}/textured_simple.obj')
        if sdf_code_source == 'gt':
            latent_code_pth = os.path.join(root_dir, '../YCB/SDF/2000/Codes/gt/%s.pth' % instance)
            recon_mesh_path = gt_mesh_path
        elif sdf_code_source == 'pred':
            latent_code_pth = os.path.join(root_dir, 'SDF/2000/Codes/pred/%s.pth' % seq_frame.replace('/', '_'))
            recon_mesh_path = latent_code_pth.replace('Codes', 'Meshes').replace('.pth', '.ply')
        else:
            raise NotImplementedError
    elif dataset_name == 'SimGrasp':
        if 'sim' not in instance:   # instance is object category
            instance = instance + '_sim'
        latent_code_dir = pjoin(root_dir, 'SDF/Reconstructions/%s/2000/Codes'%(instance))
        if sdf_code_source == 'gt':
            latent_code_pth = os.path.join(latent_code_dir, seq_frame[:5] + '.pth')
        elif sdf_code_source == 'pred':
            latent_code_pth = os.path.join(latent_code_dir, seq_frame + '.pth')
        else:
            raise NotImplementedError
        recon_mesh_path = latent_code_pth.replace('Codes', 'Meshes').replace('.pth', '.ply')
        normalization_dir = pjoin(root_dir, 'SDF/NormalizationParameters/%s'%(instance))
        normalization_pth = os.path.join(normalization_dir, seq_frame[:5] + '.npz')
        normalization_params = np.load(normalization_pth)
        saved_model_pth = pjoin(root_dir, 'SDF/examples/%s/ModelParameters/2000.pth'%(instance))
        gt_mesh_path = pjoin(root_dir, f'objs/{instance}/{seq_frame[:5]}.obj')
    elif dataset_name == 'DexYCB':
        gt_mesh_path = os.path.join(root_dir, f'../YCB/models/{instance}/textured_simple.obj')
        if sdf_code_source == 'gt':       
            latent_code_pth = os.path.join(root_dir, '../YCB/SDF/2000/Codes/gt/%s.pth' % instance)
            recon_mesh_path = gt_mesh_path
        elif sdf_code_source == 'pred':
            latent_code_pth = pjoin(root_dir, 'SDF/2000/Codes/pred/%s.pth' % seq_frame.replace('+', '_'))
            recon_mesh_path = latent_code_pth.replace('Codes', 'Meshes').replace('pred/', 'pred/%s_'%instance).replace('.pth', '.ply')
        normalization_pth = os.path.join(root_dir, '../YCB/SDF/NormalizationParameters/%s/textured_simple.npz' % instance)
        normalization_params = np.load(normalization_pth)
        if 'bowl' in instance:
            saved_model_pth = pjoin(root_dir, '../SimGrasp/SDF/examples/bowl_sim/ModelParameters/2000.pth')
            print('Use SDF decoder for bowl!')
        else:
            saved_model_pth = pjoin(root_dir, '../SimGrasp/SDF/examples/bottle_sim/ModelParameters/2000.pth')
            print('Use SDF decoder for bottle!')
    else:
        print(dataset_name)
        raise NotImplementedError
    return [latent_code_pth, normalization_params, saved_model_pth, gt_mesh_path, recon_mesh_path]


def compute_chamfer(gt_mesh, pred_mesh):
    distance_mat = torch.norm(gt_mesh.reshape(1, -1, 3) - pred_mesh.reshape(-1, 1, 3), dim=-1) # N, M
    chamfer_distance = torch.mean(torch.min(distance_mat, dim=1)[0]) + torch.mean(torch.min(distance_mat, dim=0)[0])
    return chamfer_distance

class HandTrackModel(nn.Module):
    def __init__(self, cfg, handnet=HandTrackNet, IKnet=None):
        super(HandTrackModel, self).__init__()
        print(f'[Hand Tracking] Use IKNet: {IKnet is not None}')
        # print(f"[Hand Tracking] Use shape code: {cfg['use_pred_hand_shape']}")
        print(f'[Hand Tracking] Use optimization: ', cfg['use_optimization'])

        self.use_optimization = cfg['use_optimization']
        self.sdf_code_source = cfg['sdf_code_source']
        self.sym = cfg['obj_sym']
        self.root_dir = cfg['data_cfg']['basepath']
        self.device = cfg['device']
        self.exp_folder = cfg['experiment_dir']
        self.save_folder = cfg['save_dir']
        self.data_root = cfg['root_dir']
        self.dataset_name = cfg['data_cfg']['dataset_name']
        ensure_dirs([self.save_folder])

        self.handnet = handnet(cfg)

        if IKnet is not None:
            self.IKnet = IKnet(cfg)
        else:
            self.IKnet = None
        
        if self.use_optimization:
            self.optimizer = gf_optimize_hand_pose(cfg)

        self.use_pred_obj_pose = cfg['use_pred_obj_pose']
        if self.use_pred_obj_pose:
            print('Use pred obj pose!')
        else:
            print('Use gt obj pose!')

        self.use_pred_hand_shape = cfg['use_pred_hand_shape']
        if self.use_pred_hand_shape:
            self.opt_shape = gf_optimize_hand_shape(cfg)
            print('Use opt to get hand shape code!')
        elif self.use_pred_hand_shape == False:
            print('Use gt hand shape')

        self.manolayer = OurManoLayer()

    def forward(self, input, flag_dict):
        flag_dict['track_flag'] = True
        assert flag_dict['test_flag'] == True
        if self.use_optimization:
            flag_dict['opt_flag'] = True
        else:
            flag_dict['opt_flag'] = False
        
        last_frame_kp = None
        shape_code = 0
        # initialize palm template for canonicalization
        _,canon_kp = self.manolayer(th_pose_coeffs=torch.zeros((1,48),device=self.device), 
                                    th_trans=torch.zeros((1,3),device=self.device))
        palm_template = handkp2palmkp(canon_kp)
        
        if self.use_optimization:
            obj_info = load_obj_for_opt(self.root_dir, self.dataset_name, self.sdf_code_source, input[0]['file_name'][0], input[0]['category'][0])
            self.optimizer.load_obj(obj_info[:3], input[0]['category'][0])
        
        ret_dict_lst = []
        for i, data in enumerate(input):
            data['pred_palm_template'] = palm_template
            if last_frame_kp is not None:
                # this trick is important for fast motion
                data['jittered_hand_kp'] = last_frame_kp + data['hand_points'].mean(dim=-2, keepdim=True).to(self.device).float()

            if self.IKnet is not None:
                flag_dict['IKNet_flag'] = True

                # HandTrackNet predict hand joint positions
                ret_dict = self.handnet(data, flag_dict)
                ret_dict['baseline_pred_kp'] = deepcopy(ret_dict['pred_kp'])
                data['baseline_pred_kp'] = deepcopy(ret_dict['pred_kp'])

                # optimize MANO shape code from the output of HandTrackNet
                if self.use_pred_hand_shape == 1 and i==0:    # only optimize shape code in frame 0
                    shape_code = self.opt_shape.optimize(ret_dict['baseline_pred_kp'])
                    _, kp = self.manolayer(th_pose_coeffs=torch.zeros((1,48),device=self.device), 
                                th_trans=torch.zeros((1,3),device=self.device),th_betas=shape_code)
                    palm_template = handkp2palmkp(kp)
                elif self.use_pred_hand_shape == 2 and i % 10 == 0: # update shape code every 10 frames
                    shape_code = self.opt_shape.optimize(ret_dict['baseline_pred_kp'])
                    _, kp = self.manolayer(th_pose_coeffs=torch.zeros((1,48),device=self.device), 
                                th_trans=torch.zeros((1,3),device=self.device),th_betas=shape_code)
                    palm_template = handkp2palmkp(kp)
                elif self.use_pred_hand_shape == 3 and i % 10 == 0: # optimize shape code every 10 frames, using history to improve the quality
                    shape_code = self.opt_shape.optimize(ret_dict['baseline_pred_kp'], use_old=True)
                    _, kp = self.manolayer(th_pose_coeffs=torch.zeros((1,48),device=self.device), 
                                th_trans=torch.zeros((1,3),device=self.device),th_betas=shape_code)
                    palm_template = handkp2palmkp(kp)
                elif self.use_pred_hand_shape == False and i == 0:  # use gt
                    shape_code = (data['gt_hand_pose']['mano_beta']).float().to(self.device)
                    palm_template = data['gt_hand_pose']['palm_template'].float().to(self.device)
                data['pred_beta'] = shape_code
                ret_dict['pred_beta'] = shape_code

                # IKNet to solve MANO pose code from the output of HandTrackNet
                IK_rdict = self.IKnet(data, flag_dict)
                if not self.use_optimization:
                    ret_dict['pred_kp'] = deepcopy(IK_rdict['pred_kp'])   
                ret_dict['global_pose'] = deepcopy(IK_rdict['global_pose'])
                ret_dict['MANO_theta'] = deepcopy(IK_rdict['MANO_theta']) 

                # optimize MANO pose code and hand global RT to improve HO interaction
                if self.use_optimization:
                    obj_pose = data['pred_obj_pose'] if self.use_pred_obj_pose else data['gt_obj_pose']
                    optimized_kp, optimized_mano, optimized_rot_mat, optimized_t = self.optimizer.optimize(ret_dict['MANO_theta'], ret_dict['global_pose'],
                                                     ret_dict['baseline_pred_kp'], last_frame_kp, ret_dict['pred_kp_vis_mask'], obj_pose, data['category'][0],
                                                     data['file_name'][0], shape_code, data['projection'])
                    ret_dict['pred_kp'] = optimized_kp
                    ret_dict['MANO_theta'] = optimized_mano
                    ret_dict['global_pose']['translation'] = optimized_t.unsqueeze(-1)
                    ret_dict['global_pose']['rotation'] = optimized_rot_mat.unsqueeze(0)
                # this trick is important for fast motion
                last_frame_kp = deepcopy(ret_dict['pred_kp'] - data['hand_points'].mean(dim=-2, keepdim=True).to(self.device).float())
            else:
                ret_dict = self.handnet(data, flag_dict)
                # this trick is important for fast motion
                last_frame_kp = deepcopy(ret_dict['pred_kp'] - data['hand_points'].mean(dim=-2, keepdim=True).to(self.device).float())

            if self.use_optimization and i == 0:
                ret_dict['obj_info'] = {}
                ret_dict['obj_info']['recon_path'] = obj_info[4]
                ret_dict['obj_info']['recon_scale'] = np.array([obj_info[1]['scale'][0]])
                ret_dict['obj_info']['gt_path'] = obj_info[3]
            ret_dict_lst.append(ret_dict)

        return ret_dict_lst
    
    def compute_loss(self, input, ret_dict_lst, flag_dict):
        save_flag = flag_dict['save_flag']
        flag_dict['track_flag'] = True
        assert flag_dict['test_flag'] == True

        # ret_dict_lst will update itself
        total_loss = {}
        save_dict = {}

        for i, data in enumerate(input):
            loss_dict, _ = self.handnet.compute_loss(data, ret_dict_lst[i], flag_dict)
            add_dict(total_loss, loss_dict)
            if i == 0:
                init_loss = deepcopy(loss_dict)
            for key in data['gt_obj_pose'].keys():
                data['gt_obj_pose'][key] = data['gt_obj_pose'][key].float()
            if self.use_pred_obj_pose:
                for key in data['pred_obj_pose'].keys():
                    data['pred_obj_pose'][key] = data['pred_obj_pose'][key].float()
                error_pred, _ = eval_part_full(data['gt_obj_pose'], data['pred_obj_pose'], axis=int(self.sym), up_and_down_sym=data['gt_obj_pose']['up_and_down_sym']) # B,x,x
                obj_loss = {}
                for key in error_pred:
                    obj_loss['obj_pred_'+key] = error_pred[key]
                add_dict(total_loss, obj_loss)
            if save_flag:
                cur_frame = {'gt_hand_poses':deepcopy(data['gt_hand_pose']),
                            'gt_hand_kp': deepcopy(data['gt_hand_kp']),
                             'gt_obj_poses':deepcopy(data['gt_obj_pose']),
                            'pred_hand_kp': deepcopy(ret_dict_lst[i]['pred_kp'].detach()), 
                            'file_name': deepcopy(data['file_name']),
                            'kp_error': deepcopy(loss_dict['hand_pred_kp_diff'])
                            }
                cur_frame['t_error'] = deepcopy(loss_dict['hand_pred_t_diff'])
                cur_frame['r_error'] = deepcopy(loss_dict['hand_pred_r_diff'])

                if flag_dict['IKNet_flag']:
                    global_r = mano_quat2axisang(matrix_to_unit_quaternion(ret_dict_lst[i]['global_pose']['rotation']))
                    cur_frame['pred_hand_poses'] = deepcopy({'mano_pose': torch.cat([global_r,ret_dict_lst[i]['MANO_theta']],dim=1), 
                                            'mano_trans': ret_dict_lst[i]['global_pose']['translation'].squeeze(-1),        #TODO: use OurManolayer to visualize!
                                            'mano_beta':ret_dict_lst[i]['pred_beta']
                                        })
                    cur_frame['baseline_pred_kp'] = ret_dict_lst[i]['baseline_pred_kp']
                if self.use_pred_obj_pose:
                    cur_frame['pred_obj_poses'] = deepcopy(data['pred_obj_pose'])
                # if i == 0:
                #     cur_frame['obj_info'] = ret_dict_lst[0]['obj_info']
                #     cur_frame['camera_intrinsic'] = data['projection']
                merge_dict(save_dict, cur_frame)
            
        if save_flag:
            if self.dataset_name == 'HO3D':
                save_name = input[0]['file_name'][0] + '.pkl'
                save_name = save_name.replace('/', '_')
                save_dict['CAD_ID'] = input[0]['category'][0]
                with open(pjoin(self.save_folder, save_name), 'wb') as f:
                    save_dict = cvt_numpy(save_dict)
                    pickle.dump(save_dict, f)
            elif self.dataset_name == 'HOI4D':
                save_dict['CAD_ID'] = input[0]['category'][0]

                save_name = input[0]['file_name'][0] + '.pkl'
                save_name = save_name.replace('/', '_').replace('_preprocess', '')

                with open(pjoin(self.save_folder, save_name), 'wb') as f:
                    save_dict = cvt_numpy(save_dict)
                    pickle.dump(save_dict, f)
            else:
                save_name = input[0]['category'][0] + '_' + input[0]['file_name'][0][:-4] + '.pkl'
                with open(pjoin(self.save_folder, save_name), 'wb') as f:
                    save_dict = cvt_numpy(save_dict)
                    pickle.dump(save_dict, f)
             
        valid_length = len(input)
        ret_loss = {}
        for k in total_loss.keys():
            if 'init' not in k:
                ret_loss[k] = total_loss[k] / valid_length
            else:
                ret_loss[k] = init_loss[k] 
        return ret_loss, ret_dict_lst

    def visualize(self, input, ret_dict_lst, flag_dict):
        save_flag=flag_dict['save_flag']
        flag_dict['track_flag'] = True 
        assert flag_dict['test_flag'] == True

        for i, data in enumerate(input):
            loss_dict, _ = self.handnet.compute_loss(data, ret_dict_lst[i], flag_dict)
            
            for key, value in loss_dict.items():
                print('{}: {}'.format(key, value))
            self.handnet.visualize(data, ret_dict_lst[i], flag_dict)
        return

class ObjTrackModel_Optimization(nn.Module):
    def __init__(self, cfg):
        super(ObjTrackModel_Optimization, self).__init__()
        self.exp_folder = cfg['experiment_dir']
        self.save_folder = cfg['save_dir']
        self.dataset_name = cfg['data_cfg']['dataset_name']
        self.sdf_code_source = cfg['sdf_code_source']

        self.device = cfg['device']
        ensure_dirs([self.save_folder])

        self.num_parts = cfg['num_parts']
        self.optimizer = gf_optimize_obj(cfg)
        self.sym = cfg['obj_sym']
        self.root_dir = cfg['data_cfg']['basepath']

    def forward(self, input, flag_dict):
        flag_dict['track_flag'] = True
        assert flag_dict['test_flag'] == True

        obj_info = load_obj_for_opt(self.root_dir, self.dataset_name, self.sdf_code_source, input[0]['file_name'][0], input[0]['category'][0])
        # if self.sdf_code_source == 'gt' :
        #     self.optimizer.load_obj_oracle(input[0]['category'][0], obj_info[3])
        # else:
        self.optimizer.load_obj(obj_info, input[0]['category'][0], input[0]['gt_obj_pose'], input[0]['obj_points'])

        last_frame_poses = None
        ret_dict_lst = []

        for i, data in enumerate(input):
            if last_frame_poses is not None:
                data['jittered_obj_pose'] = last_frame_poses
            else:
                data['jittered_obj_pose']['translation'] = torch.FloatTensor(data['jittered_obj_pose']['translation'].float()).reshape(1, 3, 1).to(self.device)
                data['jittered_obj_pose']['rotation'] = torch.FloatTensor(data['jittered_obj_pose']['rotation'].float()).reshape(1, 3, 3).to(self.device)
                data['jittered_obj_pose']['prev_translation'] = data['jittered_obj_pose']['translation']
                data['jittered_obj_pose']['prev_rotation'] = data['jittered_obj_pose']['rotation']
                last_frame_poses = {
                    'translation': data['jittered_obj_pose']['translation'],
                    'rotation': data['jittered_obj_pose']['rotation'],
                }
                # data['jittered_obj_pose']['scale'] = torch.FloatTensor(data['jittered_obj_pose']['scale'].float()).reshape(1,).to(self.device)

            ret_dict = self.optimizer.optimize(data['obj_points'], data['jittered_obj_pose'], data['category'][0], data['file_name'][0], data['projection'])
            last_frame_poses['prev_translation'] = deepcopy(last_frame_poses['translation']) # last frame pose
            last_frame_poses['prev_rotation'] = deepcopy(last_frame_poses['rotation'])
            last_frame_poses['translation'] = deepcopy(ret_dict['translation']) # current frame pose
            last_frame_poses['rotation'] = deepcopy(ret_dict['rotation'])

            if i == 0:
                ret_dict['scale'] = 2/obj_info[1]['scale'][0]
                ret_dict['obj_info'] = {}
                ret_dict['obj_info']['recon_path'] = obj_info[4]
                ret_dict['obj_info']['recon_scale'] = np.array([obj_info[1]['scale'][0]])
                ret_dict['obj_info']['gt_path'] = obj_info[3]
            
            ret_dict_lst.append(ret_dict)
        
        if self.optimizer.update_shape_flag:
            self.optimizer.sdf2mesh(obj_info[4].replace('.ply', '_update.ply'))

        return ret_dict_lst

    def compute_loss(self, input, ret_dict_lst, flag_dict):
        save_flag = flag_dict['save_flag']
        flag_dict['track_flag'] = True 
        assert flag_dict['test_flag'] == True
        
        # load obj info for chamfer loss
        obj_info = load_obj_for_opt(self.root_dir, self.dataset_name, self.sdf_code_source, input[0]['file_name'][0], input[0]['category'][0])
        if self.optimizer.update_shape_flag:
            mesh_filename = obj_info[4].replace('.ply', '_update.ply')
            pred_mesh = torch.FloatTensor(np.asarray(trimesh.load(mesh_filename).vertices)).to(self.device)
        else:
            pred_mesh = torch.FloatTensor(np.asarray(trimesh.load(obj_info[4]).vertices)).to(self.device)
        
        gt_mesh = trimesh.sample.sample_surface(trimesh.load(obj_info[3]), 2048)[0]
        gt_mesh = torch.FloatTensor(gt_mesh).to(self.device)
        if len(pred_mesh) > 2048:
            sample_idx = farthest_point_sample(pred_mesh, 2048, self.device)
            pred_mesh = pred_mesh[sample_idx]
        if self.sdf_code_source != 'gt':
            pred_mesh = InsCS2CatCS(pred_mesh, obj_info[1], input[0]['category'][0], self.dataset_name)
 
        # ret_dict_lst will update itself
        total_loss = {}
        save_dict = {}
        for i, data in enumerate(input):
            for key in data['gt_obj_pose'].keys():
                if key == 'scale' or key == 'up_and_down_sym':
                    continue
                data['gt_obj_pose'][key] = torch.FloatTensor(data['gt_obj_pose'][key].float()).to(self.device)
                ret_dict_lst[i][key] = ret_dict_lst[i][key][None,:]
            ret_dict_lst[i]['scale'] = np.array([1.0])
            # ret_dict_lst[i]['scale'] = np.array([ret_dict_lst[0]['scale']])
            if self.dataset_name == 'HO3D' or self.dataset_name == 'DexYCB':
                R, T  = get_RT(input[0]['category'][0])
                R = torch.FloatTensor(R).to(self.device)[None,None,:,:]
                T = torch.FloatTensor(T).to(self.device)[None,None,:,None]
                eval_gt_obj_pose = {'rotation': torch.matmul(data['gt_obj_pose']['rotation'], R.transpose(-1,-2))}
                eval_gt_obj_pose['translation'] =  data['gt_obj_pose']['translation'] - torch.matmul(eval_gt_obj_pose['rotation'], T)

                eval_pred_obj_pose = {'rotation': torch.matmul(ret_dict_lst[i]['rotation'], R.transpose(-1,-2))}
                eval_pred_obj_pose['translation'] = ret_dict_lst[i]['translation'] - torch.matmul(eval_pred_obj_pose['rotation'], T)
            else:
                eval_gt_obj_pose = data['gt_obj_pose']
                eval_pred_obj_pose = ret_dict_lst[i]
            loss_dict, _ = eval_part_full(eval_gt_obj_pose, eval_pred_obj_pose, axis=int(self.sym), up_and_down_sym=data['gt_obj_pose']['up_and_down_sym']) # B,x,x
            loss_dict['raw_obj_chamfer(mm)'] = compute_chamfer(gt_mesh, pred_mesh) * 1000
            transformed_gt_mesh = torch.matmul(gt_mesh.clone(), data['gt_obj_pose']['rotation'].squeeze().transpose(-1,-2)) + data['gt_obj_pose']['translation'].squeeze()
            transformed_pred_mesh = torch.matmul(pred_mesh.clone(), ret_dict_lst[i]['rotation'].squeeze().transpose(-1,-2)) + ret_dict_lst[i]['translation'].squeeze()
            loss_dict['pred_obj_chamfer(mm)'] = compute_chamfer(transformed_gt_mesh, transformed_pred_mesh) * 1000
            add_dict(total_loss, loss_dict)
            if save_flag:
                merge_dict(save_dict, {'gt_hand_poses':data['gt_hand_pose'],
                        'gt_obj_poses':data['gt_obj_pose'],
                         'pred_obj_poses': ret_dict_lst[i], 
                         'file_name': data['file_name'],
                         't_error_0': deepcopy(loss_dict['tdiff_0']), 
                         'r_error_0':deepcopy(loss_dict['rdiff_0']), 
                        'obj_info': ret_dict_lst[0]['obj_info'],
                        })
        if save_flag:
            if self.dataset_name == 'HO3D':
                save_name = input[0]['file_name'][0] + '.pkl'
                save_name = save_name.replace('/', '_')
                save_dict['CAD_ID'] = input[0]['category'][0]
                with open(pjoin(self.save_folder, save_name), 'wb') as f:
                    save_dict = cvt_numpy(save_dict)
                    pickle.dump(save_dict, f)
            elif self.dataset_name == 'DexYCB':
                save_name = input[0]['file_name'][0].replace('/', '_') + '.pkl'
                save_dict['CAD_ID'] = input[0]['category'][0]
                with open(pjoin(self.save_folder, save_name), 'wb') as f:
                    save_dict = cvt_numpy(save_dict)
                    pickle.dump(save_dict, f)
            elif self.dataset_name == 'HOI4D':
                save_name = input[0]['file_name'][0].replace('/', '_') + '.pkl'
                save_dict['CAD_ID'] = input[0]['category'][0]
                with open(pjoin(self.save_folder, save_name), 'wb') as f:
                    save_dict = cvt_numpy(save_dict)
                    pickle.dump(save_dict, f)
            else:
                save_name = input[0]['category'][0] + '_' + input[0]['file_name'][0][:-4] + '.pkl'
                with open(pjoin(self.save_folder, save_name), 'wb') as f:
                    save_dict = cvt_numpy(save_dict)
                    pickle.dump(save_dict, f)

        ret_loss = {}
        for k in total_loss.keys():
            if 'init' not in k:
                ret_loss[k] = total_loss[k] / len(input)

        return ret_loss, ret_dict_lst



def parse_args():
    parser = argparse.ArgumentParser('SingleFrameModel')
    parser.add_argument('--config', type=str, default='config.yml', help='path to config.yml')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

