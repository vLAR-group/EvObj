import torch
import torch.optim as optim
import os
from glob import glob
import numpy as np
import time
import torch.nn as nn
from lib.helper_ply import read_ply, write_ply
from benchmark.evaluate_semantic_instance_s3dis_chair import evaluate
import random
import colorsys
from typing import List, Tuple
import functools
import pickle
import torch.nn.functional as F
import skimage.measure
import plyfile
import logging
from torch.distributions.categorical import Categorical
from torch_scatter import scatter_mean, scatter_max, scatter_min
from collections import namedtuple
import spconv.pytorch as spconv
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'logprob', 'td_target', 'value', 'advantage'))#, 'sample_rate'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.adv = []

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            self.adv.append(None)
        self.memory[self.position] = Transition(*args)
        self.adv[self.position] = args[7].squeeze().item()
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def adv_mean_std(self):
        adv = np.array(self.adv)
        return adv.mean(), adv.std()

    def __len__(self):
        return len(self.memory)

@functools.lru_cache(20)
def get_evenly_distributed_colors(count: int) -> List[Tuple[np.uint8, np.uint8, np.uint8]]:
    HSV_tuples = [(x / count, 1.0, 1.0) for x in range(count)]
    return list(map(lambda x: (np.array(colorsys.hsv_to_rgb(*x)) * 255).astype(np.uint8),HSV_tuples))

class Trainer(object):
    def __init__(self, model, logger, val_dataset, save_path, cfg=None, use_label=False):
        self.model = model.cuda()
        self.val_dataset = val_dataset
        self.save_path = save_path
        self.logger = logger
        self.cfg = cfg
        self.use_label = use_label
        self.BATCH_SIZE = 100



    def refresh_info(self):
        ## loss
        self.loss_dict = {'loss': 0, 'ppo loss': 0 , 'actor loss': 0, 'critic loss': 0, 'ent loss': 0,
                          'seg loss': 0, 'mask loss': 0, 'dice loss': 0, 'class loss': 0}
        self.training_iter = 0
        self.step_reward = 0
        self.traj_length = 0
        self.data_time, self.optimize_time = 0, 0
        self.ious, self.ious50, self.ious25 = 0, 0, 0
        self.num_ious, self.num_ious50, self.num_ious25 = 0, 0, 0


    def load_checkpoint(self, ckpt_path=None):
        if ckpt_path is not None:
            path = ckpt_path
        else:
            checkpoints = glob(self.save_path+'/*tar')
            if len(checkpoints) == 0:
                print('No checkpoints found at {}'.format(self.save_path))
                return 0

            checkpoints = [os.path.splitext(os.path.basename(path))[0].split('_')[-1] for path in checkpoints]
            checkpoints = np.array(checkpoints, dtype=int)
            checkpoints = np.sort(checkpoints)
            path = os.path.join(self.save_path, 'checkpoint_{}.tar'.format(checkpoints[-1]))

        print('Loaded checkpoint from: {}'.format(path))
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])



        epoch = checkpoint['epoch']
        return epoch

    def validation(self, vis=True, log=False, ckpt_path=None):
        self.load_checkpoint(ckpt_path)
        self.refresh_info()
        self.preds, self.gt = {}, {}
        self.model.eval()
        val_data_loader = self.val_dataset.get_loader(shuffle=False)
        for batch_idx, batch in enumerate(val_data_loader):
            with torch.no_grad():
                coords, feature, normals, target, scene_name, semantic, instance, inverse_map, unique_map, voxl_pc, full_pc, voxl_sp, pointsp, exist_pseudo = batch
                batch_sp = [voxl_sp[i].cuda() for i in range(len(voxl_sp))]
                if not coords.is_contiguous():
                    coords = coords.contiguous()
                in_field = spconv.SparseConvTensor(features=feature.cuda(), indices=coords.int().cuda(),
                                                   spatial_shape=list(coords.max(0)[0] + 16)[1:],
                                                   batch_size=coords.max(0)[0][0].item() + 1)
                if self.cfg.use_sp:
                    output = self.model(in_field, point2segment=batch_sp, raw_coordinates=feature[:, -3:].cuda(), train_on_segments=self.cfg.use_sp)
                    sp_score = output["pred_masks"]  # [(bs), N, 10]
                    voxel_masks = sp_score[0][voxl_sp[0]].sigmoid()
                else:
                    output = self.model(in_field, raw_coordinates=feature[:, -3:].cuda(), train_on_segments=self.cfg.use_sp)
                    voxel_score = output["pred_masks"]
                    voxel_masks = voxel_score[0].sigmoid()

                masks = voxel_masks[inverse_map[0]].detach().cpu()
                hard_masks = (masks>0.5)

                valid_mask_idx, mask_score = [], []

                for mask_id in range(self.model.num_queries):
                    score = masks[:,mask_id][hard_masks[:, mask_id]].mean()
                    if torch.argmax(output["pred_logits"][0][mask_id])==0:# and (hard_masks[:, mask_id]==1).sum()>50:
                        valid_mask_idx.append(mask_id)
                        mask_score.append(score.item())  ## rec error as maskscore
                #
                valid_masks = hard_masks[:, valid_mask_idx]
                if len(valid_mask_idx)>0:
                    pred_instance_color = np.vstack(get_evenly_distributed_colors(valid_masks.shape[1]))
            if vis:
                if "auditorium" in scene_name[0]:
                    print(f"Skip visualization for {scene_name[0]} (contains 'auditorium'), otherwise may crash due to too many points.")
                    continue
                with torch.no_grad():
                    full_pc = full_pc[0].numpy()
                    non_ceiling_mask = (torch.logical_and(semantic[0]!=0, semantic[0]!=12))[inverse_map[0]]
                    area_name, room_name = scene_name[0].split('/')[0], scene_name[0].split('/')[1]
                    os.makedirs(self.cfg.save_path + '/vis/'+ area_name, exist_ok=True)
                    predcolor, gtcolor = np.ones_like(full_pc) * 128, np.ones_like(full_pc) * 128
                    for mask_id in range(valid_masks.shape[1]):
                        predcolor2 = np.ones_like(full_pc) * 128
                        mask = valid_masks[:, mask_id]
                        predcolor2[mask] = pred_instance_color[mask_id]
                        predcolor[mask] = pred_instance_color[mask_id]
                        # write_ply(os.path.join(self.cfg.save_path + '/vis', scene_name[0] + 'preds_'+str(mask_id)+'.ply'), [full_pc, predcolor2.astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])
                    pc_centered = full_pc - full_pc.mean(axis=0, keepdims=True)
                    write_ply(os.path.join(self.cfg.save_path + '/vis_s3dis/', area_name, room_name + 'preds.ply'),
                              [pc_centered[non_ceiling_mask], predcolor[non_ceiling_mask].astype(np.uint8)],
                              ['x', 'y', 'z', 'red', 'green', 'blue'])

                    if target[0]['masks'].sum()>0:
                        gt_instance_color = np.vstack(get_evenly_distributed_colors(len(target[0]['masks'])))
                        for mask_id in range(len(target[0]['masks'])):
                            gtcolor[target[0]['masks'][:, inverse_map[0]][mask_id]] = gt_instance_color[mask_id]
                    pc_centered_gt = full_pc - full_pc.mean(axis=0, keepdims=True)
                    write_ply(os.path.join(self.cfg.save_path + '/vis_s3dis/', area_name, room_name + 'gt.ply'), [pc_centered_gt[non_ceiling_mask], gtcolor[non_ceiling_mask].astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])

            self.preds[scene_name[0]] = {"pred_masks": valid_masks.cpu().numpy(), "pred_scores": (torch.tensor(mask_score)).cpu().numpy(), "pred_classes": (8+1) * torch.ones(valid_masks.shape[-1]).cpu().numpy()}
            gt_file = os.path.join(self.cfg.data_dir, 'instance_gt', scene_name[0] + '.txt')
            self.gt[scene_name[0]] = gt_file
        evaluate(self.use_label, self.preds, self.gt, self.logger, log, self.save_path)
        torch.cuda.empty_cache()
        torch.cuda.synchronize(torch.device("cuda"))


  

