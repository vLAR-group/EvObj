import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import models.obj_model as model
import models.datasets.segnet_s3dis as voxelized_data
import testing_vaeseg_s3dis
import logging
import argparse
from omegaconf import DictConfig, OmegaConf
from mask3d_spconv.sparse_unet import Custom30M
from mask3d_spconv.mask3d import Mask3D
import warnings
warnings.filterwarnings('ignore')
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (40960, rlimit[1]))

def config_parser():
    parser = argparse.ArgumentParser(description='s3dis')
    parser.add_argument("--data_dir", type=str, default='/media/SSD/zihui/simon/data/s3dis_align/processed')
    parser.add_argument("--save_path", type=str, default=os.path.join(PROJECT_ROOT, 'tests3dis', 's3dis_vae_chair'))
    parser.add_argument(
        "--test_areas",
        nargs="+",
        default=["1"],
        help="S3DIS test areas, e.g. --test_areas 1, --test_areas Area 1, 2, or --test_areas all",
    )
    parser.add_argument("--sp_dir", type=str, default='/media/SSD/zihui/simon/data/s3dis_align/SPG_0.05')
    parser.add_argument("--use_sp", type=bool, default=True)
    parser.add_argument("--cross_test_ckpt", type=str, default='./xxxx/ckpt.tar')# path of ckpt trained on ScanNet
    
    return parser.parse_args()


def parse_test_areas(test_areas):
    area_text = " ".join(test_areas)
    if area_text.strip().lower() == "all":
        return [f"Area_{area_id}" for area_id in range(1, 7)]

    area_text = area_text.replace("Area_", "").replace("Area", "")
    area_ids = [area_id.strip() for area_id in area_text.replace(",", " ").split()]

    parsed_areas = []
    for area_id in area_ids:
        if not area_id.isdigit():
            raise ValueError(f"Invalid test area '{area_id}'. Use numbers like 1 or Area_1.")
        parsed_areas.append(f"Area_{int(area_id)}")

    if not parsed_areas:
        raise ValueError("At least one test area is required.")
    return parsed_areas


def main(cfg, logger):
    '''Prepare Data'''
    # all_areas = ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_5', 'Area_6']
    test_areas = parse_test_areas(cfg.test_areas)
    logger.info(f"Testing on areas: {test_areas}")

    with open(os.path.join(PROJECT_ROOT, 'mask3d_spconv', 'mask3d_s3dis.yaml'), 'r') as file:
        model_cfg = OmegaConf.load(file)
    mask3d = mask3d_loading(model_cfg)

    val_dataset = voxelized_data.VoxelizedDataset('validation', test_areas, cfg, data_path=cfg.data_dir, batch_size=1, num_workers=4,voxel_size=0.05)

    trainer = testing_vaeseg_s3dis.Trainer(mask3d,logger, val_dataset, cfg.save_path, cfg, use_label=False)
    trainer.validation(vis=False, log=False, ckpt_path=cfg.cross_test_ckpt)


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
