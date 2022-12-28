from re import L
import yaml
import torch
import os
from os.path import join as pjoin
from utils import ensure_dirs


def overwrite_config(cfg, key, key_path, value):
    cur_key = key_path[0]
    if len(key_path) == 1:
        old_value = None if cur_key not in cfg else cfg[cur_key]
        if old_value != value:
            print("{} (originally {}) overwritten by arg {}".format(key, old_value, value))
            cfg[cur_key] = value
    else:
        if not cur_key in cfg:
            cfg[cur_key] = {}
        overwrite_config(cfg[cur_key], key, key_path[1:], value)

def choose_one_valid_path(path_lst):
    path = None 
    for i in path_lst:
        if os.path.isdir(i):
            path = i
            break
    assert path is not None, f'Candidates are {path_lst}' 
    print(path)
    return path 

def get_config(args, save=True):
    base_path = os.path.dirname(__file__)
    f = open(pjoin(base_path, 'all_config', args.config), 'r')
    cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    """ Update info from command line """
    args = vars(args)  # convert to a dictionary!
    args.pop('config')
    for key, item in args.items():
        if item is not None:
            key_path = key.split('/')
            overwrite_config(cfg, key, key_path, item)

    """ Load data config """
    f = open(pjoin(base_path, 'data_config', cfg['data_config']), 'r')
    data_cfg = yaml.load(f, Loader=yaml.FullLoader)

    """ Load pointnet config """
    point_cfg_path = pjoin(base_path, 'pointnet_config')
    pointnet_cfgs = cfg['pointnet_cfg']
    cfg['pointnet'] = {}
    for key, value in pointnet_cfgs.items():
        f = open(pjoin(point_cfg_path, value), 'r')
        cfg['pointnet'][key] = yaml.load(f, Loader=yaml.FullLoader)

    """ Save config """
    root_list = ['data']
    root = choose_one_valid_path(root_list)
    cfg['root_dir'] = root 
    if 'save_dir' not in cfg:
        cfg['save_dir'] = pjoin(root, 'exps', cfg['experiment_dir'], 'results')
    else:
        cfg['save_dir'] = pjoin(root, 'exps', cfg['save_dir'], 'results')
    cfg['experiment_dir'] = pjoin(root, 'exps', cfg['experiment_dir'])
    if 'IKNet_dir' in cfg:
        cfg['IKNet_dir'] = pjoin(root, 'exps', cfg['IKNet_dir'])
    if 'pred_obj_pose_dir' in cfg:
        cfg['pred_obj_pose_dir'] = pjoin(root, 'exps', cfg['pred_obj_pose_dir'], 'results')

    ensure_dirs(cfg['save_dir'])
    ensure_dirs(cfg['experiment_dir'])

    if save:
        yml_cfg = pjoin(cfg['experiment_dir'], 'config.yml')
        yml_obj = pjoin(cfg['experiment_dir'], cfg['data_config'])
        print('Saving config and data_config at {} and {}'.format(yml_cfg, yml_obj))
        for yml_file, content in zip([yml_cfg, yml_obj], [cfg, data_cfg]):
            with open(yml_file, 'w') as f:
                yaml.dump(content, f, default_flow_style=False)

    """ Fill in object info """
    obj_cat = cfg["obj_category"]
    if not isinstance(obj_cat, list):
        cfg["num_parts"] = data_cfg[obj_cat]["num_parts"]
        cfg["obj_sym"] = data_cfg[obj_cat]["sym"]
    else:
        cfg["num_parts"] = data_cfg[obj_cat[0]]["num_parts"]
        cfg['obj_sym'] = data_cfg[obj_cat[0]]["sym"]
        #for cat in obj_cat:
        #    assert (cfg["num_parts"] == data_cfg[cat]["num_parts"] and cfg['obj_sym'] == data_cfg[cat]["sym"]), "must use the same category!"

    cfg["data_cfg"] = data_cfg
    cfg["device"] = torch.device("cuda:%d" % cfg['cuda_id']) if torch.cuda.is_available() else "cpu"
    mano_path_lst = ['third_party/mano/models']
    cfg['mano_root'] = choose_one_valid_path(mano_path_lst)
    cfg['data_cfg']['basepath'] = pjoin(root, cfg['data_cfg']['basepath'])
    print("Running on ", cfg["device"])

    return cfg



