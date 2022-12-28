import argparse
import os
import torch
import pickle
import logging
import sys
import time
from tqdm import tqdm
import numpy as np
from copy import deepcopy
from os.path import join as pjoin
import matplotlib.pyplot as plt

base_dir = os.path.dirname(__file__)
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, '..'))

from utils import boolean_string, ensure_dirs, add_dict, log_loss_summary, print_composite
from datasets.dataset import get_dataloader
from configs.config import get_config
from trainer import Trainer
from parse_args import add_args
import time
import warnings
warnings.filterwarnings('ignore')
torch.set_num_threads(1)
def parse_args():
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    parser.add_argument('--mode_name', default='test')
    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    cfg = get_config(args, save=False)

    '''LOG'''
    log_dir = pjoin(cfg['save_dir'], '../log')
    ensure_dirs(log_dir)
    logger = logging.getLogger("TestModel")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    time_str = time.strftime("%m_%d_%H:%M", time.localtime())
    file_handler = logging.FileHandler('%s/log_test_%s.txt' % (log_dir, time_str), mode='w+')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('-' * 40 + 'Start' + '-' * 40)
    log_string('PARAMETER ...')
    log_string(cfg)

    '''testing'''
    mode_name = args.mode_name
    test_dataloader = get_dataloader(cfg, mode_name, shuffle=False)
    
    '''TRAINER'''
    trainer = Trainer(cfg, logger, test_dataloader.__len__())
    trainer.resume(test_dataloader.__len__())
    test_loss = {'cnt': 0}

    zero_time = time.time()
    time_dict = {'data_proc': 0.0, 'network': 0.0}
    total_frames = 0

    for i, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader), smoothing=0.9):
        num_frames = len(data)
        total_frames += num_frames
        log_string(f'Trajectory {i}, {num_frames:8} frames****************************')

        start_time = time.time()
        elapse = start_time - zero_time
        time_dict['data_proc'] += elapse
        print(f'Data Preprocessing: {elapse:8.2f}s {num_frames / elapse:8.2f}FPS')

        loss_dict, _ = trainer.test(data, debug=args.debug, debug_save=args.debug_save, save_flag=args.save)

        elapse = time.time() - start_time
        time_dict['network'] += elapse
        print(f'Network Forwarding: {elapse:8.2f}s {num_frames / elapse:8.2f}FPS')
        if cfg['track'] != False:
            log_string(f"File_name: {data[0]['file_name'][0]}")

        loss_dict['cnt'] = 1
        add_dict(test_loss, loss_dict)
        log_loss_summary(loss_dict, 1, lambda x, y: log_string('Test {} is {}'.format(x, y)))
        zero_time = time.time()

    log_string(f'Overall, {total_frames:8} frames****************************')
    log_string(f'Data Preprocessing: {time_dict["data_proc"]:8.2f}s {total_frames / time_dict["data_proc"]:8.2f}FPS')
    log_string(f'Network Forwarding: {time_dict["network"]:8.2f}s {total_frames / time_dict["network"]:8.2f}FPS')

    log_loss_summary(test_loss, test_loss['cnt'], lambda x, y: log_string('Test {} is {}'.format(x, y)))
    if cfg['batch_size'] > 1:
        print(f'PLEASE SET batch_size = 1 TO TEST THE SPEED. CURRENT BATCH_SIZE:', cfg["batch_size"])
    
if __name__ == '__main__':
    args = parse_args()
    torch.multiprocessing.set_start_method('spawn')
    main(args)

