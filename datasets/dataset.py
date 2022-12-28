from hashlib import sha3_512
import os
import sys
import argparse
from copy import deepcopy
from tqdm import tqdm
from os.path import join as pjoin
from torch.utils.data import Dataset, DataLoader
import numpy as np

base_dir = os.path.dirname(__file__)
sys.path.append(pjoin(base_dir, '..', '..'))
from configs.config import get_config

from SimGrasp_dataset import SimGraspDataset
from HO3D_dataset import HO3DDataset
from DexYCB_dataset import DexYCBDataset


def choose_dataset(name):
    if name == 'SimGrasp':
        return SimGraspDataset
    elif name == 'HO3D':
        return HO3DDataset
    elif name == 'DexYCB':
        return DexYCBDataset
    else:
        raise NotImplementedError

class SingleFrameData(Dataset):
    def __init__(self, cfg, mode):
        assert mode == 'train' or mode == 'test'
        self.dataset = choose_dataset(cfg['data_cfg']['dataset_name'])(cfg, mode=mode)
        self.invalid_dict = {}
        self.len = len(self.dataset)
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        ''' skip bad frame '''
        if index not in self.invalid_dict:
            data = self.dataset[index]
            if data is None:
                self.invalid_dict[index] = True
        if index in self.invalid_dict:
            return self.__getitem__((index + 1) % self.len)
        return data

class SequenceData(Dataset):
    def __init__(self, cfg, mode):
        assert mode == 'train' or mode == 'test'
        self.dataset = choose_dataset(cfg['data_cfg']['dataset_name'])(cfg, mode=mode)
        self.dataset_name = cfg['data_cfg']['dataset_name']
        if cfg['data_cfg']['dataset_name'] in ['DexYCB', 'HO3D']:
            self.seq_start = self.dataset.seq_start[:-1]
            self.seq_end = self.dataset.seq_start[1:]
            self.len = len(self.seq_start)
        elif cfg['data_cfg']['dataset_name'] in ['SimGrasp']:
            self.num_frames = cfg['data_cfg']['num_frames']
            if len(self.dataset) >= self.num_frames:
                assert len(self.dataset) % self.num_frames == 0, "Total #frames mismatch with #frames/video"
            else:
                self.num_frames = len(self.dataset)
            self.len = len(self.dataset) // self.num_frames
            self.seq_start = range(0, len(self.dataset), self.num_frames)
            self.seq_end = range(self.num_frames, len(self.dataset)+self.num_frames, self.num_frames)
        else:
            raise NotImplementedError

        print('--------------SeqenceDataset-------------------')
        print('total_len: {}\nseq_start: {}\nseq_end: {}'.format(self.len, self.seq_start,self.seq_end))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        '''
        (buggy?) Sometimes there will be a bad frame(which will return None) in a sequence.
        So far I simply use the nearest good frame after the bad one.
        '''
        seq_data = []
        for i in range(self.seq_start[idx], self.seq_end[idx]):
            data = deepcopy(self.dataset[i])
            seq_data.append(data)

        old_len = len(seq_data)
        '''add a good frame at last in avoid of that the last frame is bad'''
        for i in reversed(range(0, len(seq_data))):
            if seq_data[i] is not None:
                seq_data.append(deepcopy(seq_data[i]))
                break
        try:
            assert len(seq_data) != old_len, "all frames of a video are BAD!!!"
        except:
            return self.__getitem__((idx+1)%self.len)
        '''replace all of None with the nearest good one'''
        for i in reversed(range(0, len(seq_data)-1)):
            if seq_data[i] is None:
                seq_data[i] = deepcopy(seq_data[i+1])

        return seq_data[:-1]


def get_dataloader(cfg, mode='train', shuffle=None):
    if cfg['track']:
        batch_size = 1
        dataset = SequenceData(cfg, mode)
    else:
        batch_size = cfg['batch_size']
        dataset = SingleFrameData(cfg, mode)

    if shuffle is None:
        shuffle = (mode == 'train')
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=cfg['num_workers'])


def parse_args():
    parser = argparse.ArgumentParser('Dataset')
    parser.add_argument('--config', type=str, default='1.9_HO3D_video_test.yml', help='path to config.yml')
    parser.add_argument('--obj_config', type=str, default=None)
    parser.add_argument('--obj_category', type=str, default=None)
    parser.add_argument('--experiment_dir', type=str, default=None, help='root dir for all outputs')
    parser.add_argument('--num_points', type=int,  default=None, help='Point Number [default: 1024]')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch Size during training [default: 16]')
    parser.add_argument('--worker', type=int, default=None, help='Batch Size during training [default: 16]')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg = get_config(args, save=False)
    dataset = SequenceData(cfg, 'test', 0)
