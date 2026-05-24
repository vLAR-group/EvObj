from utils.config import *
import os
import torch


def build_comp_AdaPoinTr(ckpt_path=None):
    config = cfg_from_yaml_file('./cfgs/Shapenet_chair/AdaPoinTr.yaml')
    from tools import builder
    base_model = builder.model_builder(config.model)
 
    if ckpt_path is None:
        raise ValueError("build_comp_AdaPoinTr requires ckpt_path")
    ckpts_path = ckpt_path
    print(f"comp ckpt (AdaPoinTr): {os.path.abspath(ckpts_path)}")
    builder.load_model(base_model, ckpt_path = ckpts_path)
    base_model.cuda()

    base_model.eval()  
    print(f'build completion model (AdaPoinTr) success')
    return base_model



def build_comp_PoinTr(ckpt_path=None):
    config = cfg_from_yaml_file('./cfgs/Shapenet_chair/PoinTr.yaml')
    from tools import builder
    base_model = builder.model_builder(config.model)

    # load checkpoints
    if ckpt_path is None:
        raise ValueError("build_comp_PoinTr requires ckpt_path")
    ckpts_path = ckpt_path
    print(f"comp ckpt (PoinTr): {os.path.abspath(ckpts_path)}")
    builder.load_model(base_model, ckpt_path = ckpts_path)
    base_model.cuda()
    base_model.eval()  
    
    print(f'build completion model (PoinTr) success')
    return base_model



def build_comp_snowflake(ckpt_path=None):
    config = cfg_from_yaml_file('./cfgs/Shapenet_chair/SnowFlakeNet.yaml')
    from tools import builder
    base_model = builder.model_builder(config.model)

    if ckpt_path is None:
        raise ValueError("build_comp_SnowFlakeNet requires ckpt_path")
    ckpts_path = ckpt_path
    print(f"comp ckpt (SnowFlakeNet): {os.path.abspath(ckpts_path)}")
    builder.load_model(base_model, ckpt_path = ckpts_path)
    base_model.cuda()
    base_model.eval()  
    print(f'build completion model (SnowFlakeNet) success')
    return base_model

