
import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
    
import models.obj_model as model
import models.datasets.segnet_scannet_pp as scannetPP_dataset
import testing_scannetPP_evo
import torch
import logging
import argparse
from glob import glob
import numpy as np
from RLnet import PPO_actor, PPO_critic
from omegaconf import DictConfig, OmegaConf
# os.environ['SPCONV_ALGO'] = 'native'
# from mask3d_mink import Res16UNet18A, Mask3D, Res16UNet14, Custom30M
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
    # NOTE: For ScanNet++ loader, data_dir should be the ScanNet++ root containing:
    # - <scene_id>/scans/{mesh_aligned_0.05.ply, segments.json, segments_anno.json}
    # - metadata/semantic_classes.txt
    # - splits/nvs_sem_{train,val,test}.txt
    # ---- Defaults (edit here) ----
    DEFAULT_DATA_DIR = "/media/SSD/zihui/simon/GrabS_spconv/scannet_pp/data"
    DEFAULT_SPLITS_DIR = os.path.join(PROJECT_ROOT, "scannetpp", "splits")
    DEFAULT_CLASS_NAMES_PATH = os.path.join(PROJECT_ROOT, "scannetpp", "metadata", "semantic_classes.txt")

    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="Path to ScanNet++ root directory")
    parser.add_argument(
        "--splits_dir",
        type=str,
        default=DEFAULT_SPLITS_DIR,
        help="Path to splits folder containing nvs_sem_{train,val,test}.txt",
    )
    parser.add_argument(
        "--class_names_path",
        type=str,
        default=DEFAULT_CLASS_NAMES_PATH,
        help="Path to semantic class names txt (one class name per line)",
    )
    parser.add_argument("--save_path", type=str, default=os.path.join(PROJECT_ROOT, "segnet", "scannet_VAE_chair_tf32"))
    # Training Data Parameters
    parser.add_argument("--use_sp", type=bool, default=True)

    return parser.parse_args()

def main(cfg, logger):
    # ---- Required ScanNet++ dataset inputs (must exist) ----
    if not os.path.isdir(cfg.data_dir):
        raise FileNotFoundError(f"--data_dir not found or not a directory: {cfg.data_dir}")
    if not os.path.isdir(cfg.splits_dir):
        raise FileNotFoundError(f"--splits_dir not found or not a directory: {cfg.splits_dir}")
    if not os.path.isfile(cfg.class_names_path):
        raise FileNotFoundError(f"--class_names_path not found or not a file: {cfg.class_names_path}")
    
    mask3d_cfg_path = os.path.join(PROJECT_ROOT, "mask3d_spconv", "mask3d_scannet.yaml")
    with open(mask3d_cfg_path, 'r') as file:
        model_cfg = OmegaConf.load(file)
    mask3d = mask3d_loading(model_cfg)

    val_dataset = scannetPP_dataset.VoxelizedDataset('validation', cfg, data_path=cfg.data_dir, batch_size=1, num_workers=4, voxel_size=cfg.voxel_size)
    trainer = testing_scannetPP_evo.Trainer(mask3d, logger, val_dataset, cfg.save_path, cfg, use_label=False)

    trainer.validation(vis=False, log=False)



def mask3d_loading(model_cfg: DictConfig):
    backbone = Custom30M(in_channels=6)
    # backbone = Custom30M(in_channels=6, out_channels=model_cfg.num_classes, out_fpn=True, config=model_cfg.config.backbone.config)
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
