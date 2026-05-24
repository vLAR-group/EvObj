import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
import models.obj_model as model
import models.datasets.segnet_sys as voxelized_data
import training_synthetic_vae
import torch
import logging
import argparse
from glob import glob
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

def config_parser():
    parser = argparse.ArgumentParser(description='sys')
    parser.add_argument("--data_dir", type=str, default='/media/SSD/zihui/simon/data/synthetic_occ_5000')
    parser.add_argument("--objnet_dir", type=str, default=os.path.join(PROJECT_ROOT, 'objnet', 'multiclass'))
    parser.add_argument("--sp_dir", type=str, default='/media/SSD/zihui/simon/data/synthetic_occ_5000/SPG_0.01')
    parser.add_argument("--save_path", type=str, default=os.path.join(PROJECT_ROOT, 'segnet', 'sys_scene_vae'))

    # Training Data Parameters
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4, help='Learning rate used during training.')
    parser.add_argument("--voxel_size", type=float, default=0.05)
    parser.add_argument("--use_sp", type=bool, default=True)
    parser.add_argument("--use_norm", type=bool, default=False)
    parser.add_argument("--env_num", type=int, default=10)
    parser.add_argument("--verbose", type=bool, default=False)# print training details
    parser.add_argument("--cd_thr", type=float, default=0.10)
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
    return parser.parse_args()

def main(cfg, logger):
    '''Prepare Data'''
    mask3d_cfg_path = os.path.join(PROJECT_ROOT, 'mask3d_spconv', 'mask3d_sys.yaml')
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
    train_dataset = voxelized_data.VoxelizedDataset('train', cfg, data_path=cfg.data_dir, batch_size=cfg.batch_size, num_workers=12, voxel_size=cfg.voxel_size)
    test_RL_dataset = voxelized_data.VoxelizedDataset('test', cfg, data_path=cfg.data_dir, batch_size=1, num_workers=8, voxel_size=cfg.voxel_size, RL=True)
    test_dataset = voxelized_data.VoxelizedDataset('test', cfg, data_path=cfg.data_dir, batch_size=1, num_workers=8, voxel_size=cfg.voxel_size)
    #########################
    trainer = training_synthetic_vae.Trainer(mask3d, objnet, actor, critic, logger, train_dataset, test_dataset, test_RL_dataset, cfg.save_path, cfg, use_norm=cfg.use_norm, use_label=False)
    trainer.train_model(cfg.num_epochs)
    # trainer.validation_RL(vis=True, log=False)
    # trainer.validation(vis=True, log=False)



def mask3d_loading(model_cfg: DictConfig):
    backbone = Custom30M(in_channels=3)
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
