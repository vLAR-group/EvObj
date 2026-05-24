import argparse
import os
import sys
import time
from pathlib import Path

import torch
from tensorboardX import SummaryWriter

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from training_code.tools import run_net, test_net
from utils import dist_utils, misc
from utils.logger import get_root_logger
from utils.config import get_config, log_args_to_file, log_config_to_file, cfg_from_yaml_file


def parse_args(root_dir: str):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='yaml config file')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--deterministic', action='store_true', help='set deterministic CUDNN backend')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')

    parser.add_argument('--exp_name', type=str, default='default', help='experiment name')
    parser.add_argument('--start_ckpts', type=str, default=None, help='reload used ckpt path')
    parser.add_argument('--ckpts', type=str, default=None, help='test used ckpt path')
    parser.add_argument('--val_freq', type=int, default=1, help='validation frequency')
    parser.add_argument('--resume', action='store_true', default=False, help='auto resume training')
    parser.add_argument('--test', action='store_true', default=False, help='test mode for ckpt')
    parser.add_argument('--dataset_cfg', type=str, default=None, help='override train/val/test dataset cfg yaml')

    args = parser.parse_args()

    if args.test and args.resume:
        raise ValueError('--test and --resume cannot be both enabled')
    if args.resume and args.start_ckpts is not None:
        raise ValueError('--resume and --start_ckpts cannot be both enabled')
    if args.test and args.ckpts is None:
        raise ValueError('ckpts should not be None while test mode')

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.test:
        args.exp_name = 'test_' + args.exp_name

    cfg_path = Path(args.config)
    args.experiment_path = os.path.join(root_dir, 'experiments', cfg_path.stem, cfg_path.parent.stem, args.exp_name)
    args.tfboard_path = os.path.join(root_dir, 'experiments', cfg_path.stem, cfg_path.parent.stem, 'TFBoard', args.exp_name)
    args.log_name = cfg_path.stem

    os.makedirs(args.experiment_path, exist_ok=True)
    os.makedirs(args.tfboard_path, exist_ok=True)

    return args


def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    os.chdir(root_dir)

    args = parse_args(root_dir)

    args.use_gpu = torch.cuda.is_available()
    if args.use_gpu:
        torch.backends.cudnn.benchmark = True

    if args.launcher == 'none':
        args.distributed = False
        world_size = 1
    else:
        args.distributed = True
        dist_utils.init_dist(args.launcher)
        _, world_size = dist_utils.get_dist_info()
        args.world_size = world_size

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, name=args.log_name)

    train_writer = None
    val_writer = None
    if not args.test and args.local_rank == 0:
        train_writer = SummaryWriter(os.path.join(args.tfboard_path, 'train'))
        val_writer = SummaryWriter(os.path.join(args.tfboard_path, 'test'))

    config = get_config(args, logger=logger)

    if args.dataset_cfg is not None:
        dataset_cfg = cfg_from_yaml_file(args.dataset_cfg)
        config.dataset.train._base_ = dataset_cfg
        config.dataset.val._base_ = dataset_cfg
        config.dataset.test._base_ = dataset_cfg
        logger.info(f'Override dataset cfg by {args.dataset_cfg}')

    if args.distributed:
        assert config.total_bs % world_size == 0
        config.dataset.train.others.bs = config.total_bs // world_size
    else:
        config.dataset.train.others.bs = config.total_bs

    log_args_to_file(args, 'args', logger=logger)
    log_config_to_file(config, 'config', logger=logger)
    logger.info(f'Distributed training: {args.distributed}')

    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, deterministic: {args.deterministic}')
        misc.set_random_seed(args.seed + args.local_rank, deterministic=args.deterministic)

    if args.test:
        test_net(args, config)
    else:
        run_net(args, config, train_writer, val_writer)


if __name__ == '__main__':
    main()
