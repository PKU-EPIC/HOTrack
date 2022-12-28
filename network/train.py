import argparse
import os
import torch
import logging
import sys
import time
from os.path import join as pjoin
import tqdm
import cv2

base_dir = os.path.dirname(__file__)
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, '..'))

from utils import ensure_dirs, add_dict, log_loss_summary, print_composite, tensorboard_logger
from datasets.dataset import get_dataloader
from configs.config import get_config
from trainer import Trainer
from parse_args import add_args
from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    cfg = get_config(args)

    '''LOG'''
    log_dir = pjoin(cfg['experiment_dir'], 'log')
    ensure_dirs(log_dir)
    #tensorboardx
    writer = SummaryWriter(log_dir)

    logger = logging.getLogger("TrainModel")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/log.txt' % (log_dir))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(cfg)

    '''DATA'''
    train_dataloader = get_dataloader(cfg, 'train', shuffle=True)
    test_dataloader = get_dataloader(cfg, 'test')

    '''TRAINER'''
    trainer = Trainer(cfg, logger, train_dataloader.__len__())
    start_epoch = trainer.resume(train_dataloader.__len__())

    for epoch in tqdm.tqdm(range(start_epoch, cfg['total_epoch'])):
        trainer.step_epoch()

        # -------------------- train ---------------------------
        train_loss = {}
        for i, data in tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc='train'):
            loss_dict = trainer.update(data, debug=args.debug, debug_save=args.debug_save)

            loss_dict['cnt'] = 1
            add_dict(train_loss, loss_dict)

        cnt = train_loss.pop('cnt')
        log_loss_summary(train_loss, cnt, lambda x, y: log_string('Train {} is {}'.format(x, y)))
        tensorboard_logger(writer, epoch, train_loss, cnt, 'Train')

        if (epoch + 1) % cfg['freq']['save'] == 0:
            trainer.save()

        # -------------------- test ---------------------------
        test_loss = {}
        for i, data in tqdm.tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc='test'):
            loss_dict, ret_dict = trainer.test(data, debug=args.debug, debug_save=args.debug_save)
            loss_dict['cnt'] = 1
            add_dict(test_loss, loss_dict)

        cnt = test_loss.pop('cnt')
        log_loss_summary(test_loss, cnt, lambda x, y: log_string('Test {} is {}'.format(x, y)))
        tensorboard_logger(writer, epoch, test_loss, cnt, 'Test')
        
if __name__ == '__main__':
    args = parse_args()
    torch.multiprocessing.set_start_method('spawn')
    main(args)

