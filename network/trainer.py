import argparse
import os
import logging
import math
import time
from os.path import join as pjoin

from collections import OrderedDict
from matplotlib.pyplot import flag
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.optim import lr_scheduler
from models.hand_network import *
from models.track_network import *

from utils import update_dict, ensure_dirs


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
    return init_fun


def get_scheduler(optimizer, cfg, dataset_len, it=-1):
    scheduler = None
    if optimizer is None:
        return scheduler
    if 'lr_policy' not in cfg or cfg['lr_policy'] == 'constant':
        scheduler = None  # constant scheduler
    elif cfg['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer,
                                        step_size=cfg['lr_step_size'],
                                        gamma=cfg['lr_gamma'],
                                        last_epoch=it)
    elif cfg['lr_policy'] == 'CyclicLR':
        step_size = (cfg['total_epoch'] * dataset_len) // 6
        scheduler = lr_scheduler.CyclicLR(
            optimizer, base_lr=5e-5, max_lr=5e-4, cycle_momentum=False,
            step_size_up= step_size,
            mode='triangular'
        )
        print('using cyclic learning rate schedule, step size is %d' % step_size)
    else:
        assert 0, '{} not implemented'.format(cfg['lr_policy'])
    return scheduler


def get_optimizer(params, cfg):
    if len(params) == 0:
        return None
    if cfg['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(
            params, lr=cfg['learning_rate'],
            betas=(0.9, 0.999), eps=1e-08,
            weight_decay=cfg['weight_decay'])
    elif cfg['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(
            params, lr=cfg['learning_rate'],
            momentum=0.9)
    else:
        assert 0, "Unsupported optimizer type {}".format(cfg['optimizer'])
    return optimizer


def get_last_model(dirname, key=""):
    if not os.path.exists(dirname):
        return None
    models = [pjoin(dirname, f) for f in os.listdir(dirname) if
              os.path.isfile(pjoin(dirname, f)) and
              key in f and ".pt" in f]
    if models is None or len(models) == 0:
        return None
    models.sort()
    last_model_name = models[-1]
    return last_model_name


def get_model_name(type_name, track):
    if type_name == 'HandTrackNet':     # HandTrackNet
        model = HandTrackNet
    elif type_name == 'iknet':      # IKNet 
        model = IKNet
    else:
        raise NotImplementedError
    return model

class Trainer(nn.Module):
    def __init__(self, cfg, logger=None, dataset_len=None):
        super(Trainer, self).__init__()
        self.ckpt_dir = pjoin(cfg['experiment_dir'], 'ckpt')
        ensure_dirs(self.ckpt_dir)
        self.loss_weights = cfg['network']['loss_weight'] if 'loss_weight' in cfg['network'].keys() else {}
        self.device = cfg['device']

        if cfg['track'] == 'hand':
            self.model = HandTrackModel(cfg, handnet=get_model_name(cfg['network']['type'], track=True))
            self.coord_exp_dir = None
            self.rotnet_exp_dir = None
            self.hand_kp_exp_dir = pjoin(cfg['experiment_dir'], 'ckpt')
            self.hand_kp_resume_epoch = cfg['resume_epoch'] if 'resume_epoch' in cfg else -1
            self.IKNet_exp_dir = None
        elif cfg['track'] == 'hand_IKNet':
            self.model = HandTrackModel(cfg, handnet=get_model_name(cfg['network']['type'], track=True), IKnet=IKNet)
            self.coord_exp_dir = None
            self.rotnet_exp_dir = None
            self.hand_kp_exp_dir = pjoin(cfg['experiment_dir'], 'ckpt')
            self.hand_kp_resume_epoch = -1
            self.IKNet_exp_dir = pjoin(cfg['IKNet_dir'], 'ckpt')
            self.IKNet_resume_epoch = -1
        elif cfg['track'] == 'obj_opt':
            self.model = ObjTrackModel_Optimization(cfg)
            self.rotnet_exp_dir = None
            self.coord_exp_dir = None
            self.hand_kp_exp_dir = None
            self.IKNet_exp_dir = None
        else:
            # for training single model
            self.model = get_model_name(cfg['network']['type'], track=False)(cfg)
            self.rotnet_exp_dir = None
            self.coord_exp_dir = None
            self.hand_kp_exp_dir = None
            self.IKNet_exp_dir = None
            self.optimizer = get_optimizer([p for p in self.model.parameters() if p.requires_grad], cfg)
            self.scheduler = get_scheduler(self.optimizer, cfg, dataset_len=dataset_len)
        
        self.warm_up = cfg['warm_up'] / 100 * cfg['total_epoch']
        self.apply(weights_init(cfg['weight_init']))
        self.epoch = 0
        self.iteration = 0
        self.cfg = cfg
        self.logger = logger
        self.to(self.device)

    def log_string(self, str):
        print(str)
        if self.logger is not None:
            self.logger.info(str)

    def summarize_losses(self, loss_dict):
        total_loss = 0
        for key, item in self.loss_weights.items():
            if key in loss_dict:
                total_loss += loss_dict[key] * item
            else:
                print('There is no loss term called {}!'.format(key))
        loss_dict['total_loss'] = total_loss
        return loss_dict

    def step_epoch(self):
        cfg = self.cfg
        self.epoch += 1
        if self.epoch < self.warm_up:
            self.lr = self.epoch * cfg['learning_rate'] / self.warm_up
        else:
            if self.scheduler is not None and self.scheduler.get_lr()[0] > cfg['lr_clip']:
                self.scheduler.step()
            self.lr = self.scheduler.get_lr()[0]

        self.log_string("Epoch %d/%d, learning rate = %f" % (
            self.epoch, cfg['total_epoch'], self.lr))

        momentum = cfg['momentum_original'] * (
                cfg['momentum_decay'] ** (self.epoch // cfg['momentum_step_size']))
        momentum = max(momentum, cfg['momentum_min'])
        self.log_string("BN momentum updated to %f" % momentum)
        self.momentum = momentum

        def bn_momentum_adjust(m, momentum):
            if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
                m.momentum = momentum

        self.model = self.model.apply(lambda x: bn_momentum_adjust(x, momentum))

    def resume(self, dataset_len):
        def get_model(dir, resume_epoch):
            if resume_epoch > 0:
                specified_model = pjoin(dir, f"model_{resume_epoch:04d}.pt")
                if os.path.exists(specified_model):
                    last_model_name = specified_model
            else:
                last_model_name = get_last_model(dir)
            print('last model name', last_model_name)
            
            return last_model_name

        ckpt = OrderedDict()

        if self.hand_kp_exp_dir is not None:
            hand_name = get_model(self.hand_kp_exp_dir, self.hand_kp_resume_epoch)
            if hand_name is None:
                assert 0, 'Invalid HandNet dir'
            else:
                print(f'Load HandKPNet model from {hand_name}')
            hand_state_dict = torch.load(hand_name, map_location=self.device)['model']
            hand_keys = list(hand_state_dict.keys())
            for key in hand_keys:
                ckpt['handnet.' + key] = hand_state_dict[key]

        
        if self.IKNet_exp_dir is not None:
            hand_name = get_model(self.IKNet_exp_dir, self.IKNet_resume_epoch)
            if hand_name is None:
                assert 0, 'Invalid IKNet dir'
            else:
                print(f'Load IKNet model from {hand_name}')
            hand_state_dict = torch.load(hand_name, map_location=self.device)['model']
            hand_keys = list(hand_state_dict.keys())
            for key in hand_keys:
                    ckpt['IKnet.' + key] = hand_state_dict[key]
        
        if not self.cfg['track']:
            model_name = get_model(self.ckpt_dir, self.cfg['resume_epoch'])

            if model_name is None:
                self.log_string('Initialize from 0')
            else:
                state_dict = torch.load(model_name, map_location=self.device)
                self.epoch = state_dict['epoch']
                self.iteration = state_dict['iteration']
                ckpt.update(state_dict['model'])

                if self.optimizer is not None:
                    try:
                        self.optimizer.load_state_dict(state_dict['optimizer'])
                    except (ValueError, KeyError):
                        pass  # when new params are added, just give up on the old states
                    self.scheduler = get_scheduler(self.optimizer, self.cfg, it=self.epoch, dataset_len=dataset_len)

            self.log_string('Resume from epoch %d' % self.epoch)

        self.model.load_state_dict(ckpt, strict=False)

        return self.epoch

    def save(self, name=None, extra_info=None):
        epoch = self.epoch
        if name is None:
            name = f'model_{epoch:04d}'
        savepath = pjoin(self.ckpt_dir, "%s.pt" % name)
        state = {
            'epoch': epoch,
            'iteration': self.iteration,
            'model': self.model.state_dict(),
        }
        state['optimizer'] = self.optimizer.state_dict()

        if isinstance(extra_info, dict):
            update_dict(state, extra_info)
        torch.save(state, savepath)
        self.log_string("Saving model at epoch {}, path {}".format(epoch, savepath))
    
    def init_flag_dict(self):
        flag_dict = {}
        flag_dict['track_flag'] = False
        flag_dict['save_flag'] = False 
        flag_dict['test_flag'] = False 
        flag_dict['IKNet_flag'] = False
        return flag_dict

    def update(self, data, debug=False, debug_save=False):
        self.model.train()

        self.optimizer.zero_grad()
        flag_dict = self.init_flag_dict()
        ret_dict = self.model(data, flag_dict)
        loss_dict, ret_dict = self.model.compute_loss(data, ret_dict, flag_dict)
        loss_dict = self.summarize_losses(loss_dict)
        loss_dict['total_loss'].backward()
        self.optimizer.step()

        self.iteration += 1

        if debug or debug_save:
            for key, value in loss_dict.items():
                print('{}: {}'.format(key, value))
            flag_dict = self.init_flag_dict()
            if debug_save:
                flag_dict['save_flag'] = True 
                self.model.visualize(data, ret_dict, flag_dict)
            else:
                self.model.visualize(data, ret_dict, flag_dict)

        loss_dict['learning_rate'] = self.lr
        return loss_dict

    def test(self, data, debug=False, debug_save=False, save_flag=False):
        flag_dict = self.init_flag_dict()
        flag_dict['test_flag'] = True
        flag_dict['save_flag'] = save_flag

        if self.cfg['track'] != 'obj_opt':
            self.model.eval()
            with torch.no_grad():
                ret_dict = self.model(data, flag_dict)
                loss_dict, ret_dict = self.model.compute_loss(data, ret_dict, flag_dict)
        else:
            self.model.train()
            ret_dict = self.model(data, flag_dict)
            loss_dict, ret_dict = self.model.compute_loss(data, ret_dict, flag_dict)            

        if debug or debug_save:
            for key, value in loss_dict.items():
                print('{}: {}'.format(key, value))
            print()
            flag_dict = self.init_flag_dict()
            flag_dict['test_flag'] = True
            if debug_save:
                flag_dict['save_flag'] = True 
                self.model.visualize(data, ret_dict, flag_dict)
            else:
                self.model.visualize(data, ret_dict, flag_dict)
        return loss_dict, ret_dict

