import os
import sys
import argparse
import logging
from glob import glob

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import models.obj_model as model
import models.datasets.segnet_scannet as voxelized_data
import training_scannet_vae
import torch
import numpy as np
from RLnet import PPO_actor, PPO_critic
from omegaconf import DictConfig, OmegaConf
from mask3d_spconv.sparse_unet import Custom30M
from mask3d_spconv.mask3d import Mask3D
import warnings
warnings.filterwarnings('ignore')
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (40960, rlimit[1]))
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def config_parser():
    parser = argparse.ArgumentParser(description='scannet')
    parser.add_argument("--data_dir", type=str, default='/media/SSD/zihui/simon/data/scannet/processed/')
    parser.add_argument("--objnet_dir", type=str, default=os.path.join(PROJECT_ROOT, 'objnet', 'chair'))
    parser.add_argument("--sp_dir", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=os.path.join(PROJECT_ROOT, 'segnet', 'scannet_vae'))

    # Training Data Parameters
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=600)
    parser.add_argument("--lr", type=float, default=1e-4, help='Learning rate used during training.')
    parser.add_argument("--voxel_size", type=float, default=0.05)
    parser.add_argument("--use_sp", type=bool, default=True)
    parser.add_argument("--use_norm", type=bool, default=False)
    parser.add_argument("--env_num", type=int, default=50)
    parser.add_argument("--verbose", type=bool, default=False)# print training details
    parser.add_argument("--cd_thr", type=float, default=0.10, help='CD threshold for reward computation')

    # discern & comp & evo
    parser.add_argument(
        "--discern_backend",
        type=str,
        default="sparseUnet",
        choices=["sparseUnet", "pointTransformer", "pointNet"],
        help="Discerning network backend.",
    )
    parser.add_argument(
        "--discern_net_ckpt",
        type=str,
        default=None,
        help="Checkpoint path for discerning backend; if unset, backend default is used.",
    )
    parser.add_argument(
        "--comp_backend",
        type=str,
        default="AdaPoinTr",
        choices=["AdaPoinTr", "PoinTr", "SnowFlakeNet"],
        help="Completion backend."
    )
    parser.add_argument(
        "--compnet_ckpt",
        type=str,
        default=None,
        help="Optional checkpoint path for selected completion backend."
    )
    # EVO 
    parser.add_argument("--evo_T", type=int, default=100, help='Evo period in epochs (e.g., T=100 -> 100,200,300,...)')
    return parser.parse_args()

def main(cfg, logger):
    if cfg.evo_T <= 0:
        raise ValueError(f"evo_T must be > 0, got {cfg.evo_T}")
    evo_epochs = list(range(cfg.evo_T, cfg.num_epochs + 1, cfg.evo_T))
    # evo_epochs = list(range(0, cfg.num_epochs + 1, cfg.evo_T)) # for debug
    cfg.evo_epochs = evo_epochs
    print(f"Discerning Evolve will occur every {cfg.evo_T} epochs at: {evo_epochs}")
    
    mask3d_cfg_path = os.path.join(PROJECT_ROOT, 'mask3d_spconv', 'mask3d_scannet.yaml')
    with open(mask3d_cfg_path, 'r') as file:
        model_cfg = OmegaConf.load(file)
    mask3d = mask3d_loading(model_cfg)

    objnet = model.PointNet2_wpos().eval().cuda()
    objnet_checkpoints = glob(os.path.join(cfg.objnet_dir, 'vae') + '/*tar')
    objnet_checkpoints = [os.path.splitext(os.path.basename(path))[0].split('_')[-1] for path in objnet_checkpoints]
    objnet_checkpoints = np.array(objnet_checkpoints, dtype=int)
    objnet_checkpoints = np.sort(objnet_checkpoints)
    path = os.path.join(os.path.join(cfg.objnet_dir, 'vae'), 'checkpoint_{}.tar'.format(objnet_checkpoints[-1]))
    print('Loaded checkpoint from: {}'.format(path))
    objnet.load_state_dict(torch.load(path)['model_state_dict'])

    n_actions = [4 + 1, 2 + 1]
    actor = PPO_actor(n_actions).cuda()
    critic = PPO_critic().cuda()
    #########################
    train_dataset = voxelized_data.VoxelizedDataset('train', cfg, data_path=cfg.data_dir, batch_size=cfg.batch_size, num_workers=8, voxel_size=cfg.voxel_size)
    val_dataset = voxelized_data.VoxelizedDataset('validation', cfg, data_path=cfg.data_dir, batch_size=1, num_workers=4, voxel_size=cfg.voxel_size)
    #########################
    trainer = training_scannet_vae.Trainer(
        mask3d, objnet, actor, critic, logger, train_dataset, val_dataset,
        cfg.save_path, cfg, use_norm=cfg.use_norm, use_label=False,
        discern_voxel_size=cfg.voxel_size,
    )
    trainer.train_model(cfg.num_epochs)
    # trainer.validation(vis=False, log=False)
    # trainer.save_scannet_for_online_test()



def mask3d_loading(model_cfg: DictConfig):
    backbone = Custom30M(in_channels=6)
    relevant_params = {key: value for key, value in model_cfg.items() if key in Mask3D.__init__.__code__.co_varnames}
    mask3d = Mask3D(backbone, **relevant_params)
    return mask3d

def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Logging to a file
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(file_handler)
    # Logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)
    return logger


if __name__ == '__main__':
    # Record start time and print start timestamp
    import time
    from datetime import datetime
    
    start_time = time.time()
    start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Training started at: {start_timestamp}")
    
    cfg = config_parser()

    '''Setup logger'''
    if not os.path.exists(cfg.save_path):
        os.makedirs(cfg.save_path)
    logger = set_logger(os.path.join(cfg.save_path, 'train.log'))
    # #
    os.system(f"cp {__file__} {cfg.save_path}")
    os.system(f"cp -r {os.path.join(PROJECT_ROOT, 'models')} {cfg.save_path}")
    os.system(f"cp -r {os.path.join(PROJECT_ROOT, 'mask3d_spconv')} {cfg.save_path}")
    os.system(f"cp {os.path.join(PROJECT_ROOT, 'RLnet.py')} {cfg.save_path}")

    main(cfg, logger)
    
    # Record end time and calculate total duration
    end_time = time.time()
    end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_seconds = int(end_time - start_time)
    
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    
    print(f"Training finished at: {end_timestamp}")
    print(f"Total training time: {hours:02d}h {minutes:02d}m {seconds:02d}s")