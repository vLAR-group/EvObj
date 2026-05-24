import torch
import torch.optim as optim
import os
from glob import glob
import numpy as np
import time
import torch.nn as nn
from lib.helper_ply import read_ply, write_ply
import spconv.pytorch as spconv
from mask3d_spconv.matcher_tmp import HungarianMatcher
from benchmark.evaluate_semantic_instance_scannetPP_multi_cls import evaluate
import random
import colorsys
from typing import List, Tuple
import functools
import torch.nn.functional as F
import skimage.measure
import plyfile
import logging
from collections import namedtuple
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'logprob', 'td_target', 'value', 'advantage'))#, 'sample_rate'))
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(ROOT, 'completion'))
sys.path.insert(1, ROOT)
sys.path.insert(2, os.path.join(ROOT, 'segment_spconv'))


def save_ply_with_mask(points: torch.Tensor, mask: torch.Tensor, filepath: str):
    """
    Save point cloud with mask information as PLY file for training
    Args:
        points: [N, 3] point coordinates
        mask: [N] binary mask (0=background, 1=foreground)
        filepath: output PLY file path
    """
    try:
        from plyfile import PlyData, PlyElement
        import numpy as np
        
        points_np = points.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy().astype(np.uint8)
        
        # Create structured array with coordinates and mask
        vertices = np.array([tuple(list(p) + [int(m)]) for p, m in zip(points_np, mask_np)], 
                           dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('mask', 'u1')])
        
        vertex_element = PlyElement.describe(vertices, 'vertex')
        ply_data = PlyData([vertex_element], text=True)
        ply_data.write(filepath)
        # print(f"Saved PLY with mask: {filepath}")
        
    except ImportError:
        print(f"Warning: plyfile not available, cannot save PLY with mask: {filepath}")
    except Exception as e:
        print(f"Error saving PLY with mask: {e}")
    
    
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
        self.batch_iter = 2
        self.mask_min_size = 100


    def refresh_info(self):
        ## loss
        self.loss_dict = {'loss': 0, 'ppo loss': 0 , 'actor loss': 0, 'critic loss': 0, 'ent loss': 0,
                          'seg loss': 0, 'mask loss': 0, 'dice loss': 0, 'class loss': 0, 'evo seg loss': 0}
        self.training_iter = 0
        self.step_reward = 0
        self.traj_length = 0
        self.data_time, self.optimize_time = 0, 0
        self.infer_time = 0
        self.ious, self.ious50, self.ious25 = 0, 0, 0
        self.num_ious, self.num_ious50, self.num_ious25 = 0, 0, 0
        self.evo_sample_counter = 0

    def _gt_ids_from_sem_inst(self, semantic_voxel, instance_voxel, inverse_map):
        """
        Build ScanNet-style gt_ids (class_id*1000 + instance_id) for evaluation, but from dataloader outputs.

        IMPORTANT: `semantic` from ScanNet++ loader is 0-based index into `class_names`.
        To keep ScanNet evaluator conventions safe (and avoid the "<1000 group" special-case),
        we shift class ids to 1-based and instance ids to 1-based in the encoding:
          class_id = semantic + 1
          instance_id = instance + 1
        Ignore points keep gt_ids = -1.

        Inputs are voxel-level semantic/instance (shape [N_voxel]) and inverse_map (len N_full).
        """
        sem = semantic_voxel.detach().cpu().numpy().astype(np.int64, copy=False)
        inst = instance_voxel.detach().cpu().numpy().astype(np.int64, copy=False)
        inv = np.asarray(inverse_map, dtype=np.int64)

        sem_full = sem[inv]
        inst_full = inst[inv]

        gt_ids = -np.ones_like(sem_full, dtype=np.int64)
        valid = (sem_full != -1) & (inst_full != -1)
        gt_ids[valid] = (sem_full[valid] + 1) * 1000 + (inst_full[valid] + 1)
        return gt_ids

    def _pred_class_ids_by_gt_vote(self, semantic_voxel, inverse_map, pred_masks_full):
        """
        Assign a class id to each predicted mask using GT semantic majority vote (like your ScanNet multi-class eval).

        Returns class ids in the same 1-based space as `_gt_ids_from_sem_inst`: (semantic + 1).
        """
        sem = semantic_voxel.detach().cpu().numpy().astype(np.int64, copy=False)
        inv = np.asarray(inverse_map, dtype=np.int64)
        sem_full = sem[inv]  # (N_full,)

        K = pred_masks_full.shape[1]
        out = np.full((K,), -1, dtype=np.int64)
        for k in range(K):
            mask = pred_masks_full[:, k].astype(bool)
            if mask.sum() == 0:
                continue
            vals = sem_full[mask]
            vals = vals[vals != -1]
            if vals.size == 0:
                continue
            # majority vote
            cls = int(np.bincount(vals.astype(np.int64)).argmax())
            out[k] = cls + 1
        return out

    def load_checkpoint(self):
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

    def validation(self, vis=True, log=False):
        self.load_checkpoint()
        self.refresh_info()
        self.preds, self.gt = {}, {}
        self.preds_cls = {}
        self.model.eval()
        val_data_loader = self.val_dataset.get_loader(shuffle=False)
        for batch_idx, batch in enumerate(val_data_loader):
            with torch.no_grad():
                coords, feature, normals, target, scene_name, semantic, instance, inverse_map, unique_map, voxl_pc, full_pc, voxl_sp, pointsp, voxl_mysp, exist_pseudo = batch
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
                    if torch.argmax(output["pred_logits"][0][mask_id])==0 and hard_masks[:, mask_id].sum()>self.mask_min_size:
                        valid_mask_idx.append(mask_id)
                        mask_score.append(score.item())  ## rec error as maskscore
                #
                valid_masks = hard_masks[:, valid_mask_idx]
                if len(valid_mask_idx)>0:
                    pred_instance_color = np.vstack(get_evenly_distributed_colors(valid_masks.shape[1]))

            if vis and len(valid_mask_idx)>0:
                with torch.no_grad():
                    full_pc = full_pc[0].numpy()
                    os.makedirs(self.cfg.save_path + '/vis_scannetpp', exist_ok=True)
                    predcolor, gtcolor = np.ones_like(full_pc) * 128, np.ones_like(full_pc) * 128
                    for mask_id in range(valid_masks.shape[1]):
                        predcolor2 = np.ones_like(full_pc) * 128
                        mask = valid_masks[:, mask_id]
                        predcolor2[mask] = pred_instance_color[mask_id]
                        predcolor[mask] = pred_instance_color[mask_id]
                    pc_centered = full_pc - full_pc.mean(axis=0, keepdims=True)
                    write_ply(
                        os.path.join(self.cfg.save_path + '/vis_scannetpp', scene_name[0] + 'input.ply'),
                        [pc_centered],
                        ['x', 'y', 'z']
                    )
                    write_ply(os.path.join(self.cfg.save_path + '/vis_scannetpp', scene_name[0] + 'pred_EVO.ply'), [pc_centered, predcolor.astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])

                    if len(target[0]['masks'])>0:
                        gt_instance_color = np.vstack(get_evenly_distributed_colors(len(target[0]['masks'])))
                    for mask_id in range(len(target[0]['masks'])):
                        gtcolor[target[0]['masks'][:, inverse_map[0]][mask_id]==1] = gt_instance_color[mask_id]
                    # centerize
                    gt_centered = full_pc - full_pc.mean(axis=0, keepdims=True)
                    write_ply(os.path.join(self.cfg.save_path + '/vis_scannetpp', scene_name[0] + 'gt.ply'), [gt_centered, gtcolor.astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])

            pred_masks_full = valid_masks.cpu().numpy()
            pred_scores = (torch.tensor(mask_score)).cpu().numpy()
            self.preds[scene_name[0]] = {
                "pred_masks": pred_masks_full,
                "pred_scores": pred_scores,
                "pred_classes": 5 * torch.ones(pred_masks_full.shape[-1]).cpu().numpy(),
            }
            gt_ids = self._gt_ids_from_sem_inst(semantic[0], instance[0], inverse_map[0])
            self.gt[scene_name[0]] = gt_ids
            cls_ids = self._pred_class_ids_by_gt_vote(semantic[0], inverse_map[0], pred_masks_full)
            self.preds_cls[scene_name[0]] = {
                "pred_masks": pred_masks_full,
                "pred_scores": pred_scores,
                "pred_classes": cls_ids,
            }
        # class-agnostic
        evaluate(False, self.preds, self.gt, class_names=getattr(self.val_dataset, "class_names", None), logger=self.logger, log=log, prcurv_save_dir=self.save_path)
        # per-class
        evaluate(True, self.preds_cls, self.gt, class_names=getattr(self.val_dataset, "class_names", None), logger=self.logger, log=log, prcurv_save_dir=self.save_path)
        torch.cuda.empty_cache()
        torch.cuda.synchronize(torch.device("cuda"))

