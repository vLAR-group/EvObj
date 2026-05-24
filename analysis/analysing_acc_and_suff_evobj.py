import open3d as o3d
from torch.utils.data import DataLoader, Dataset
from collections import namedtuple
from torch_scatter import scatter_mean
from torch.distributions.categorical import Categorical
import logging
import plyfile
import skimage.measure
import torch.nn.functional as F
import pickle
import functools
from typing import List, Tuple
import colorsys
import random
from benchmark.evaluate_semantic_instance_scannet_chair import evaluate
from mask3d_spconv.matcher_tmp import HungarianMatcher
import spconv.pytorch as spconv
from lib.helper_ply import write_ply
import torch.nn as nn
import time
import numpy as np
from glob import glob
import torch.optim as optim
import torch
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if os.path.join(ROOT, 'completion') not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, 'completion'))
if ROOT not in sys.path:
    sys.path.insert(1, ROOT)
if os.path.join(ROOT, 'segment_spconv') not in sys.path:
    sys.path.insert(2, os.path.join(ROOT, 'segment_spconv'))
if os.path.join(ROOT, 'segmentation') not in sys.path:
    sys.path.append(os.path.join(ROOT, 'segmentation'))

# Default discerning net weights (used when cfg.discern_net_ckpt is unset).
_DISC_MODULE_ROOT = os.path.join(ROOT, 'discerning_module')
DEFAULT_DISCERNING_NET_CKPT = {
    "sparseUnet": os.path.join(_DISC_MODULE_ROOT, "ckpts", "spUnet_scannet.pth"),
    "pointTransformer": os.path.join(
        _DISC_MODULE_ROOT, "ckpts", "pointTransformer_scannet.pth"),
    "pointNet": os.path.join(_DISC_MODULE_ROOT, "ckpts", "pointNet_scannet.pth"),
}

# Default completion net weights (used when cfg.compnet_ckpt is unset).
_COMP_MODULE_ROOT = os.path.join(ROOT, 'completion_module')
DEFAULT_COMP_NET_CKPT = {
    "AdaPoinTr": os.path.join(_COMP_MODULE_ROOT, "ckpts", "AdaPoinTr_scannet_ckpt.pth"),
    "PoinTr": os.path.join(_COMP_MODULE_ROOT, "ckpts", "PoinTr_scannet_ckpt.pth"),
    "SnowFlakeNet": os.path.join(_COMP_MODULE_ROOT, "ckpts", "Snowflakenet_scannet_ckpt.pth"),
}

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward',
                        'logprob', 'td_target', 'value', 'advantage'))  # , 'sample_rate'))


def build_comp_wrapper(comp_backend="AdaPoinTr", checkpoint_path=None):
    original_cwd = os.getcwd()
    ckpt = checkpoint_path
    if ckpt is None or (isinstance(ckpt, str) and not str(ckpt).strip()):
        ckpt = DEFAULT_COMP_NET_CKPT.get(comp_backend)
        if ckpt is None:
            raise ValueError(
                f'comp_backend must be "AdaPoinTr", "PoinTr", or "SnowFlakeNet", '
                f'got {comp_backend!r}')
        print('[CKPT] using default completion checkpoint: {}'.format(ckpt))
    os.chdir(_COMP_MODULE_ROOT)
    sys.path.insert(0, _COMP_MODULE_ROOT)
    if comp_backend == "AdaPoinTr":
        from build_comp import build_comp_AdaPoinTr
        comp_model = build_comp_AdaPoinTr(ckpt_path=ckpt)
    elif comp_backend == "PoinTr":
        from build_comp import build_comp_PoinTr
        comp_model = build_comp_PoinTr(ckpt_path=ckpt)
    elif comp_backend == "SnowFlakeNet":
        from build_comp import build_comp_snowflake
        comp_model = build_comp_snowflake(ckpt_path=ckpt)
    else:
        os.chdir(original_cwd)
        raise ValueError(
            f'comp_backend must be "AdaPoinTr", "PoinTr", or "SnowFlakeNet", got {comp_backend!r}')

    print(f"build comp_{comp_backend} success")
    os.chdir(original_cwd)
    return comp_model


def build_discern_wrapper(discern_backend="sparseUnet", checkpoint_path=None,
                      in_channels=3, num_classes=2):
    from discerning_module.build_discern import DiscerningModuleBuilder
    ckpt = checkpoint_path
    if ckpt is None or (isinstance(ckpt, str) and not str(ckpt).strip()):
        ckpt = DEFAULT_DISCERNING_NET_CKPT.get(discern_backend)
        print('[CKPT] using default discerning checkpoint: {}'.format(ckpt))
    if discern_backend == "sparseUnet":
        discern_net = DiscerningModuleBuilder.sparse_unet(ckpt)
    elif discern_backend == "pointTransformer":
        discern_net = DiscerningModuleBuilder.point_transformer(
            in_channels=in_channels,
            num_classes=num_classes,
            checkpoint_path=ckpt,
        )
    elif discern_backend == "pointNet":
        discern_net = DiscerningModuleBuilder.point_net(ckpt)
    else:
        raise ValueError(
            f'discern_backend must be "sparseUnet", "pointTransformer", or "pointNet", '
            f'got {discern_backend!r}')
    print(f"build discern_{discern_backend} success")
    return discern_net

def save_ply_with_mask(points: torch.Tensor, mask: torch.Tensor, filepath: str):
    try:
        from plyfile import PlyData, PlyElement
        import numpy as np

        points_np = points.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy().astype(np.uint8)
        colors = np.zeros((points_np.shape[0], 3), dtype=np.uint8)
        colors[:] = np.array([128, 128, 128], dtype=np.uint8)
        colors[mask_np == 1] = np.array([255, 0, 0], dtype=np.uint8)
        vertices = np.array(
            [tuple(list(p) + list(c)) for p, c in zip(points_np, colors)],
            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                   ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        )
        vertex_element = PlyElement.describe(vertices, 'vertex')
        ply_data = PlyData([vertex_element], text=True)
        ply_data.write(filepath)
    except ImportError:
        print(
            f"Warning: plyfile not available, cannot save PLY with mask: {filepath}")
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
    return list(map(lambda x: (np.array(colorsys.hsv_to_rgb(*x)) * 255).astype(np.uint8), HSV_tuples))


class EvoDatasetFromFiles(Dataset):
    """Dataset for loading evo data from PLY files with mask information"""

    def __init__(self, data_dir, num_point=4096, voxel_size=0.05, cfg=None):
        self.data_dir = data_dir
        self.num_point = num_point
        self.voxel_size = voxel_size
        self.ply_files = self._find_ply_files()
        print(f"Found {len(self.ply_files)} evo data files in {data_dir}")

    def _find_ply_files(self):
        ply_files = []
        if os.path.exists(self.data_dir):
            for file in os.listdir(self.data_dir):
                if file.endswith('.ply') and 'input_with_labels' in file:
                    ply_files.append(os.path.join(self.data_dir, file))
        return ply_files

    def __len__(self):
        return len(self.ply_files)

    def __getitem__(self, idx):
        ply_file = self.ply_files[idx]
        pcd = o3d.io.read_point_cloud(ply_file)
        points = np.asarray(pcd.points)  # [N, 3].
        masks = self._read_mask_from_ply(ply_file)

        # Sample points if needed
        if len(points) > self.num_point:
            indices = np.random.choice(
                len(points), self.num_point, replace=False)
            points = points[indices]
            masks = masks[indices]
        elif len(points) < self.num_point:
            # Pad with random points if too few
            pad_size = self.num_point - len(points)
            pad_indices = np.random.choice(len(points), pad_size, replace=True)
            points = np.vstack([points, points[pad_indices]])
            masks = np.hstack([masks, masks[pad_indices]])

        # Apply augmentation.
        points = self.random_z_rotate(points)
        points = self.random_scale(points)

        return torch.from_numpy(points).float(), torch.from_numpy(masks).long()

    def random_z_rotate(self, points):
        """Randomly rotate points around the z-axis"""
        theta = np.random.uniform(0, 2 * np.pi)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        rot_mat = np.array([
            [cos_theta, -sin_theta, 0],
            [sin_theta,  cos_theta, 0],
            [0,          0,         1]
        ], dtype=np.float32)
        return points @ rot_mat.T

    def random_scale(self, points):
        """Randomly scale points by factor in range [0.9, 1.1]"""
        scale_factor = np.random.uniform(0.9, 1.1)
        return points * scale_factor

    def _read_mask_from_ply(self, ply_file):
        ply_data = plyfile.PlyData.read(ply_file)
        vertex = ply_data['vertex']
        prop_names = set(vertex.data.dtype.names)
        if {'red', 'green', 'blue'}.issubset(prop_names):
            red = np.asarray(vertex['red'], dtype=np.uint8)
            green = np.asarray(vertex['green'], dtype=np.uint8)
            blue = np.asarray(vertex['blue'], dtype=np.uint8)
            masks = ((red > 200) & (green < 80) & (blue < 80)).astype(np.int64)
            return masks
        if 'mask' in prop_names:
            return np.asarray(vertex['mask'], dtype=np.int64)
        raise ValueError(
            f"PLY file {ply_file} does not contain RGB or mask fields for evo labels.")


class Trainer(object):
    def __init__(self, model, objnet, PPO_actor, PPO_critic, logger, train_dataset, val_dataset, save_path, cfg=None, use_norm=True, use_label=False, discern_voxel_size=0.05, val_RL_dataset=None):
        self.model = model.cuda()
        self.objnet = objnet.cuda().eval()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=cfg.lr)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.val_RL_dataset = val_RL_dataset if val_RL_dataset is not None else val_dataset
        self.save_path = save_path
        self.logger = logger
        self.cfg = cfg
        self.discern_backend = getattr(self.cfg, "discern_backend", None) or "sparseUnet"
        self.current_epoch = 0 
        self.use_label = use_label
        self.use_norm = use_norm
        self.topk_query = model.num_queries
        self.matcher = HungarianMatcher()
        self.BATCH_SIZE = 100
        self.GAMMA = 0.900
        self.n_actions = [4+1, 2+1]
        self.max_step = 8
        self.max_eval_step = 8
        self.actor, self.critic = PPO_actor, PPO_critic
        self.optimizer_actor = optim.Adam(
            self.actor.parameters(), lr=1e-4, eps=1e-5)
        self.optimizer_critic = optim.Adam(
            self.critic.parameters(), lr=1e-4, eps=1e-5)

        self.alpha = 0.2  # in [0, 1]. Scaling factor
        self.nu = 10  # Reward of Trigger
        self.threshold = 0.5
        self.clip_actor_eps = 0.2
        self.gae_lambda = 0.5
        self.gae = True
        self.ent_coeff = 0.1
        self.clip_value = False
        self.clip_value_eps = 0.1
        self.normalize_adv = True
        self.anchor_env_r = 2.0  
        self.moving_step = 0.3

        self.obj_r = 0.6
        self.obj_h = 1.8
        self.scene_h = 4
        self.min_obj_r = 0.15  # use EFEM param 
        self.max_obj_r = 0.6
        self.max_obj_h = 1.8

        self.self_atten_sample_num = 1024
        self.convergence_sample_num = 1024
        self.obj_center_z = 0.6

        self.batch_iter = 2
        self.sp_pseudo = True
        self.traj_dict_capa = 8*self.cfg.env_num
        self.initial_R = 1.5
        self.min_R = 0.15
        self.max_R = 1.5
        self.R_decay = 0.75
        self.sdf_bs = 100

        self.cd_thr = self.cfg.cd_thr
        print(f'setting cd_thr: {self.cd_thr}')
        self.discern_voxel_size = discern_voxel_size
        print(f'setting discern_voxel_size: {self.discern_voxel_size}')
        self.mask_min_size = 100
        self.bcyl_min_size = 50

        self.scaler = torch.cuda.amp.GradScaler()

        self.comp_backend = self.cfg.comp_backend if hasattr(
            self.cfg, "comp_backend") else "AdaPoinTr"
        self.comp_net = build_comp_wrapper(
            comp_backend=self.comp_backend,
            checkpoint_path=getattr(self.cfg, "compnet_ckpt", None),
        )
        print(f"[COMPLETION] Trainer comp_backend={self.comp_backend}")

        self.discern_net = build_discern_wrapper(
            discern_backend=self.discern_backend,
            checkpoint_path=getattr(self.cfg, "discern_net_ckpt", None),
        )
        print(f"[DISCERNING] Trainer discern_backend={self.discern_backend}")

        self.optimizer_discern = optim.Adam(
            self.discern_net.parameters(), lr=1e-4, weight_decay=1e-4)

        # Evolution settings
        self.evo_batch_size = 32  # 32
        self.evo_epochs_per_batch = 40
        self.evo_thr = self.cfg.cd_thr
        self.evo_lr = 1e-3
        self.evo_sample_counter = 0  # Counter for naming saved files
        self.evo_epochs = self.cfg.evo_epochs # Epochs to collect and evolve discerning module
        print(f'[EVOLUTION] setting evo_thr: {self.evo_thr}')
        print(f'[EVOLUTION] setting evo_epochs: {self.evo_epochs}')

        # IoU thresholds for sufficiency statistics
        self.sufficiency_iou_thresholds = getattr(
            self.cfg, 'iou_thresholds', [0.5, 0.6, 0.7, 0.8]
        )
        print(f'setting sufficiency_iou_thresholds: {self.sufficiency_iou_thresholds}')

    def refresh_info(self):
        # loss
        self.loss_dict = {'loss': 0, 'ppo loss': 0, 'actor loss': 0, 'critic loss': 0, 'ent loss': 0,
                          'seg loss': 0, 'mask loss': 0, 'dice loss': 0, 'class loss': 0, 'evo discern loss': 0}
        self.training_iter = 0
        self.logging_interval = len(
            self.train_dataset.get_loader(shuffle=True))*self.batch_iter
        self.step_reward = 0
        self.traj_length = 0
        self.data_time, self.optimize_time = 0, 0
        self.infer_time = 0
        self.ious, self.ious50, self.ious25 = 0, 0, 0
        self.num_ious, self.num_ious50, self.num_ious25 = 0, 0, 0
        self.evo_sample_counter = 0

    def init_traj_dict(self):
        self.traj_dict = {}

    def assign_env_info(self, traj_id, cur_bs, cur_env, all_actions, history, curpos, curR, initial_bcyl_center, env_feature, env_xyz, env_norm, cur_GT_bcyl_mask, mask_completeness):
        initial_bcy_mask = self.compute_bcyl(
            [], env_xyz, curpos, initial_bcyl_center, self.initial_R)[1]  # replaced
        if initial_bcy_mask.sum() > self.bcyl_min_size:
            self.traj_dict[str(traj_id)] = {}

            self.traj_dict[str(traj_id)]['bcyl_mask'] = initial_bcy_mask
            self.traj_dict[str(traj_id)]['cur_bs'] = cur_bs  # fixed
            self.traj_dict[str(traj_id)]['cur_env'] = cur_env  # fixed
            # accumulated
            self.traj_dict[str(traj_id)]['all_actions'] = all_actions
            self.traj_dict[str(traj_id)]['history'] = history  # accumulated
            self.traj_dict[str(traj_id)]['curpos'] = curpos  # replaced
            self.traj_dict[str(traj_id)]['curR'] = curR  # replaced
            # fixed
            self.traj_dict[str(traj_id)
                           ]['initial_bcyl_center'] = initial_bcyl_center
            self.traj_dict[str(traj_id)]['env_feature'] = env_feature  # fixed
            self.traj_dict[str(traj_id)]['env_xyz'] = env_xyz  # fixed
            self.traj_dict[str(traj_id)]['env_norm'] = env_norm  # fixed
            # fixed
            self.traj_dict[str(traj_id)]['cur_GT_bcyl_mask'] = cur_GT_bcyl_mask
            self.traj_dict[str(traj_id)]['done'] = [False]  # increased
            self.traj_dict[str(traj_id)]['traj'] = []  # increased
            # replaced
            self.traj_dict[str(traj_id)]['dist2target'] = self.anchor_env_r
            # replaced
            self.traj_dict[str(traj_id)]['target_mask_center'] = None
            self.traj_dict[str(traj_id)]['W'] = torch.zeros_like(
                env_xyz)[:, 0]  # .cpu() ### replaced
            self.traj_dict[str(traj_id)]['iou'] = 0
            self.traj_dict[str(traj_id)
                           ]['mask_completeness'] = mask_completeness

    def train_batch(self, batch, batch_idx, epoch, loader_size):
        self.current_epoch = epoch
        time_cur = time.time()
        coords, feature, normals, target, scene_name, semantic, instance,  inverse_map,  unique_map, voxl_pc, pointpc, voxl_sp, pointsp, voxl_mysp, exist_pseudo = batch
        batch_sp, pc = [voxl_sp[i].cuda() for i in range(len(voxl_sp))], [
            voxl_pc[i].cuda() for i in range(len(voxl_sp))]
        if not coords.is_contiguous():
            coords = coords.contiguous()
        in_field = spconv.SparseConvTensor(features=feature.cuda(), indices=coords.int().cuda(
        ), spatial_shape=list(coords.max(0)[0] + 16)[1:], batch_size=coords.max(0)[0][0].item()+1)
        bs = len(pc)
        pseudo_batch, obj_score = [[] for _ in range(len(pc))], [
            [] for _ in range(len(pc))]  # batch size of []
        pseudo_bcyl_batch = [[] for _ in range(len(pc))]

        # 1. compute point/sp features and mask, these are only used to compute the TD_target, so can be no gradien
        self.model.eval(), self.actor.eval(), self.critic.eval()
        with torch.no_grad():
            if self.cfg.use_sp:
                output = self.model(in_field, point2segment=batch_sp, raw_coordinates=feature[:, -3:].cuda(
                ), train_on_segments=self.cfg.use_sp, env_num=self.cfg.env_num, is_datacollect=True)
            else:
                output = self.model(in_field, raw_coordinates=feature[:, -3:].cuda(
                ), train_on_segments=self.cfg.use_sp, env_num=self.cfg.env_num, is_datacollect=True)
            bkb_feature = [f.detach() for f in output["mask_features"]]
            batch_anchor = output['sampled_coords'].detach()  # [bs, K, 3]

        self.infer_time += time.time() - time_cur
        time_cur = time.time()

        # 2. collect trajectory for current batch
        ###### for each batch, we have B scene, each scene has 50 anchor, leading to 50*B trajectory ########
        self.memory, step_R, step_num, traj_num = ReplayMemory(
            bs*self.cfg.env_num*self.max_step), 0, 0, 0
        # Here we represent state by the point_idx
        with torch.no_grad():
            state_index, state_pred_mask = [], []
            # to measure the quality of mask
            for b in range(bs):
                state_index.append([]), state_pred_mask.append([])
                pc2anchor = (pc[b][:, None, :] - batch_anchor[b]
                             [None, ...])[:, :, 0:2].norm(p=2, dim=-1)  # [x, y]
                for anchor_idx in range(self.cfg.env_num):
                    in_env_idx = torch.where(
                        pc2anchor[:, anchor_idx] <= self.anchor_env_r)[0]
                    if len(in_env_idx) >= 500:
                        state_index[-1].append(in_env_idx)
                    else:
                        state_index[-1].append(None)
                    state_pred_mask[-1].append(None)

        # split traj id into sets
        traj_id_set = []
        for traj_id in range(bs*self.cfg.env_num):
            if traj_id % self.traj_dict_capa == 0:
                traj_id_set.append([])
            traj_id_set[-1].append(traj_id)

        # random sample 10% traj, otherwise the data collection in RL is too time-consuming
        for l in range(len(traj_id_set)):
            traj_id_set[l] = np.random.choice(traj_id_set[l], len(
                traj_id_set[l])//10, replace=False).tolist()

        for traj_ids in traj_id_set:
            # traj dict is only an temporary storage, it mainly used to record the state feature for many traj
            self.init_traj_dict()
            # 2.1 init some environment info for current traj_id_set to dict
            for traj_id in traj_ids:
                cur_bs, cur_env = traj_id//self.cfg.env_num, traj_id % self.cfg.env_num
                if state_index[cur_bs][cur_env] is None:
                    continue

                all_actions, history, bcyl_center = [], torch.zeros(
                    (self.max_step, 6 + 1)).cuda(), batch_anchor[cur_bs][cur_env].unsqueeze(0)
                ##
                curpos, initial_bcyl_center = bcyl_center.clone(), bcyl_center.clone()
                curpos[:, -1], initial_bcyl_center[:, -
                                                   1] = curpos[:, -1]*0, initial_bcyl_center[:, -1]*0
                env_feature = bkb_feature[cur_bs][state_index[cur_bs][cur_env]]
                env_xyz = pc[cur_bs][state_index[cur_bs][cur_env]]
                env_norm = normals[cur_bs][state_index[cur_bs][cur_env]]

                GT_mask = target[cur_bs]['masks'].cuda()  # [K', N]
                GT_env_mask = GT_mask[:, state_index[cur_bs][cur_env]]
                env_GTmask_ratio = GT_env_mask.sum(-1)/GT_mask.sum(-1)
                ###
                GT_env_thr = 0.01  # if this env has taeget object, record its mask, GT here is only to check the training process, not influence training
                if max(env_GTmask_ratio) >= GT_env_thr:
                    traj_num += 1
                    GT_idx = torch.where(env_GTmask_ratio >= GT_env_thr)[0]
                    cur_GT_env_mask = GT_env_mask[GT_idx].t()  # [N, K]
                    if len(cur_GT_env_mask.shape) == 1:
                        cur_GT_env_mask = cur_GT_env_mask.unsqueeze(-1)
                    cur_GT_bcyl_mask = cur_GT_env_mask
                    # now we know this is an valud traj_id, so assign to dict
                    self.assign_env_info(traj_id, cur_bs, cur_env, all_actions, history, curpos, self.initial_R,
                                         initial_bcyl_center, env_feature, env_xyz, env_norm, cur_GT_bcyl_mask, env_GTmask_ratio[GT_idx])
                else:
                    # no valid GT in cur area
                    traj_num += 1
                    self.assign_env_info(traj_id, cur_bs, cur_env, all_actions, history, curpos,
                                         self.initial_R, initial_bcyl_center, env_feature, env_xyz, env_norm, None, None)

            # 2.2 making steps simultaneously for the current traj_id_set, using dict
            # 2.2.1 compute state, state feature, action, logprob, value for these set
            for t in range(self.max_step):
                state_list, not_done_traj = [], []
                cur_envfeat_list, cur_hist_list, cur_centered_pos, cur_centered_envxyz, cur_env_feats, cur_bcyl_mask = [], [], [], [], [], []
                cur_inbcyl_xyz, cur_inbcyl_feats, cur_R = [], [], []
                for traj_id in self.traj_dict.keys():
                    if len(self.traj_dict[traj_id]['done']) > t:
                        if not self.traj_dict[traj_id]['done'][t]:
                            # state: bcylindar center, history, batch_id, anchor_id
                            state = (self.traj_dict[traj_id]['curpos'], self.traj_dict[traj_id]['curR'], self.traj_dict[traj_id]['history'].unsqueeze(0),
                                     self.traj_dict[traj_id]['cur_bs'], self.traj_dict[traj_id]['cur_env'], self.traj_dict[traj_id]['initial_bcyl_center'])
                            state_list.append(state)
                            not_done_traj.append(traj_id)
                            ###
                            cur_centered_pos.append(
                                (self.traj_dict[traj_id]['curpos'] - self.traj_dict[traj_id]['initial_bcyl_center']).unsqueeze(0))
                            ###
                            cur_centered_envxyz.append(
                                (self.traj_dict[traj_id]['env_xyz'] - self.traj_dict[traj_id]['initial_bcyl_center']).unsqueeze(0))
                            cur_env_feats.append(
                                self.traj_dict[traj_id]['env_feature'].unsqueeze(0))
                            # tiny mask3d
                            cur_bcyl_mask.append(
                                self.traj_dict[traj_id]['bcyl_mask'].unsqueeze(0))
                            inbcyl_xyz = (self.traj_dict[traj_id]['env_xyz'] - self.traj_dict[traj_id]['initial_bcyl_center'])[
                                torch.where(self.traj_dict[traj_id]['bcyl_mask'])[0]]
                            inbcyl_feats = self.traj_dict[traj_id]['env_feature'][torch.where(
                                self.traj_dict[traj_id]['bcyl_mask'])[0]]
                            cur_R.append(self.traj_dict[traj_id]['curR'])

                            sample_idx = np.random.choice(inbcyl_xyz.shape[0], self.self_atten_sample_num, replace=False) if inbcyl_xyz.shape[0] >= self.self_atten_sample_num \
                                else np.random.choice(inbcyl_xyz.shape[0], self.self_atten_sample_num, replace=True)
                            cur_inbcyl_xyz.append(inbcyl_xyz[sample_idx].unsqueeze(
                                0)), cur_inbcyl_feats.append(inbcyl_feats[sample_idx].unsqueeze(0))
                            cur_hist_list.append(
                                self.traj_dict[traj_id]['history'].unsqueeze(0))

                if len(not_done_traj) == 0:
                    break
                else:
                    cur_centered_pos = torch.cat(cur_centered_pos)
                    cur_centered_pos[:, :, -1] *= 0

                    cur_inbcyl_xyz, cur_inbcyl_feats, cur_history = torch.cat(
                        cur_inbcyl_xyz), torch.cat(cur_inbcyl_feats), torch.cat(cur_hist_list)
                    actions, logprobs, values, state_feats = self.select_action(
                        cur_inbcyl_xyz, torch.tensor(cur_R), cur_inbcyl_feats, cur_centered_pos, cur_history)

                    # record bcyl_mask
                    for idx, traj_id in enumerate(not_done_traj):
                        action = actions[idx].unsqueeze(0)
                        self.traj_dict[traj_id]['all_actions'].append(action)
                        bcyl_center, bcyl_mask, curR = self.compute_bcyl(action, self.traj_dict[traj_id]['env_xyz'], self.traj_dict[traj_id]['curpos'],
                                                                         self.traj_dict[traj_id]['initial_bcyl_center'], self.traj_dict[traj_id]['curR'])
                        self.traj_dict[traj_id]['curpos'] = bcyl_center
                        self.traj_dict[traj_id]['bcyl_mask'] = bcyl_mask
                        self.traj_dict[traj_id]['curR'] = curR

                    # only for compute reward, when we take the action, what reward can we have?
                    sdfmask, inmask_pc, inmask_norm, inmask_prob = self.compute_sdfmask(
                        not_done_traj, batch_sp=batch_sp, state_index=state_index)
                    if inmask_pc is not None:
                        valid_mask, pc2mesh, mesh2pc = self.compute_convergence(
                            sdfmask, inmask_pc, inmask_prob,
                            save_dir=os.path.join(self.save_path, f"vis_comp/step_{t:03d}")
                        )  # only for compute reward, when we take the action, what reward can we have?
                                                                           

                    for idx, traj_id in enumerate(not_done_traj):
                        action, logprob, value = actions[idx].unsqueeze(
                            0), logprobs[idx], values[idx]
                        if inmask_pc is not None:
                            reward, pc2mesh_inrange_ratio, mesh2pc_inrange_ratio = self.compute_reward_CD(
                                idx, valid_mask, pc2mesh, mesh2pc,
                                traj_id=traj_id, sdfmask=sdfmask[idx] if sdfmask is not None else None)
                        else:
                            reward = -1

                        # use GT to check
                        if self.traj_dict[traj_id]['cur_GT_bcyl_mask'] is not None:
                            iou = self.get_maxmatch_mask(
                                self.traj_dict[traj_id]['cur_GT_bcyl_mask'], self.traj_dict[traj_id]['mask_completeness'], sdfmask[idx]).item()
                        else:
                            iou = 0  # None

                        if reward == self.nu:
                            if self.cfg.verbose:
                                print('IoU with GT:', iou, 'pc2mesh distance:',
                                      pc2mesh_inrange_ratio, 'mesh2pc distance:', mesh2pc_inrange_ratio)
                            self.ious += iou
                            self.num_ious += 1
                            if iou >= 0.25:
                                self.ious25 += iou
                                self.num_ious25 += 1
                            if iou >= 0.5:
                                self.ious50 += iou
                                self.num_ious50 += 1
                            next_state = None
                            done = True
                        else:
                            next_state = (self.traj_dict[traj_id]['curpos'], self.traj_dict[traj_id]['curR'], self.traj_dict[traj_id]['history'].unsqueeze(
                                0), self.traj_dict[traj_id]['cur_bs'], self.traj_dict[traj_id]['cur_env'])
                            done = False
                            ##
                            reward = -1

                        if t == self.max_step-1:
                            done = True
                        self.traj_dict[traj_id]['traj'].append(
                            (state_list[idx], action, next_state, reward, logprob, value, done))
                        self.traj_dict[traj_id]['done'].append(done)

                        # pseudo mask, load and save
                        if reward == self.nu:
                            point_pseudo = torch.zeros_like(
                                pc[self.traj_dict[traj_id]['cur_bs']])[:, 0]
                            point_pseudo[state_index[self.traj_dict[traj_id]['cur_bs']]
                                         [self.traj_dict[traj_id]['cur_env']]] = sdfmask[idx]  
                            point_bcyl = torch.zeros_like(
                                pc[self.traj_dict[traj_id]['cur_bs']])[:, 0]
                            point_bcyl[state_index[self.traj_dict[traj_id]['cur_bs']]
                                       [self.traj_dict[traj_id]['cur_env']]] = self.traj_dict[traj_id]['bcyl_mask'].float()
                            ranking_score = - \
                                (pc2mesh_inrange_ratio +
                                 mesh2pc_inrange_ratio).item()+10
                            if self.cfg.use_sp:
                                pseudo_batch[self.traj_dict[traj_id]['cur_bs']].append(point_pseudo.unsqueeze(
                                    -1)), obj_score[self.traj_dict[traj_id]['cur_bs']].append(torch.tensor(ranking_score).unsqueeze(0))
                                pseudo_bcyl_batch[self.traj_dict[traj_id]['cur_bs']].append(point_bcyl.unsqueeze(-1))
                            else:
                                pseudo_batch[self.traj_dict[traj_id]['cur_bs']].append(point_pseudo.unsqueeze(
                                    -1)), obj_score[self.traj_dict[traj_id]['cur_bs']].append(torch.tensor(ranking_score).unsqueeze(0))
                                pseudo_bcyl_batch[self.traj_dict[traj_id]['cur_bs']].append(point_bcyl.unsqueeze(-1))

                        step_R += reward
                        step_num += 1

            for traj_id in list(self.traj_dict.keys()):
                if self.traj_dict[traj_id]['traj'][-1][3] != self.nu:
                    if random.random() <= 0.9 and len(list(self.traj_dict.keys())) > 10:
                        del self.traj_dict[traj_id]
            # 2.2.1 compute state, state feature, action, logprob, value for these set
            # generalizaed advantage
            print_reward = True
            if print_reward:
                for traj_id in self.traj_dict.keys():
                    trajectory = self.traj_dict[traj_id]['traj']
                    tmp_reward_list = []
                    for t in range(len(trajectory)):
                        tmp_reward_list.append(trajectory[t][3])

            for traj_id in self.traj_dict.keys():
                trajectory = self.traj_dict[traj_id]['traj']
                if self.gae:
                    lastgae, gae = 0, torch.zeros(len(trajectory))
                    for t in reversed(range(len(trajectory))):
                        if t == len(trajectory) - 1:
                            next_done = trajectory[-1][-1]
                            # not exist next state trajectory[-1][-2]
                            next_value = 0
                            nextnonterminal = 1.0 - next_done
                            nextvalues = next_value
                        else:
                            nextnonterminal = 1.0 - trajectory[t][-1]
                            nextvalues = trajectory[t + 1][-2]
                        td_delta = trajectory[t][3] + self.GAMMA * \
                            nextvalues * nextnonterminal - trajectory[t][-2]
                        gae[t] = lastgae = td_delta + self.GAMMA * \
                            self.gae_lambda * nextnonterminal * lastgae

                for step, (state, action, next_state, reward, logprob, value, done) in enumerate(trajectory):
                    # bootstrap value if not done
                    if step < len(trajectory) - 1:
                        next_value = trajectory[step + 1][-2]
                    else:
                        next_value = 0
                    if self.gae:
                        advantage = gae[step]
                        td_target = value + gae[step]
                    else:
                        td_target = reward + self.GAMMA * \
                            next_value * (1 - done)
                        td_delta = td_target - value
                        advantage = td_delta
                    self.memory.push(state, action, next_state,
                                     reward, logprob, td_target, value, advantage)

        torch.cuda.empty_cache()
        torch.cuda.synchronize(torch.device("cuda"))
        del output

        # pseudo mask, load and save, only this phase will take 50seconds
        # sometimes masks in pseudo_mask_list are sp mask
        pseudo_mask_list, valid_bs, gt_mask_list = [], [], []
        for b in range(len(pc)):
            # First, count GT for this scene (independent of whether there are pseudo masks)
            scene_gt_count = 0
            GT_mask = None
            if target[b]['masks'].shape[0] > 0:
                GT_mask = target[b]['masks'].cuda()  # [K_gt, N]
                scene_gt_count = GT_mask.shape[0]
                # New and Merged use the same GT count (all scenes' GT)
                self.epoch_new_pseudo_gt += scene_gt_count
                self.epoch_merged_pseudo_gt += scene_gt_count

            # Record number of new pseudo masks collected in this batch
            num_new_pseudo = len(pseudo_batch[b]) if len(pseudo_batch[b]) > 0 else 0

            if len(pseudo_batch[b]) > 0:
                # add exist pseudo to current
                # Separate new pseudo masks and existing pseudo masks
                new_pseudo = torch.cat(pseudo_batch[b], dim=-1)  # [M, K_new]
                new_bcyl = torch.cat(pseudo_bcyl_batch[b], dim=-1).float()  # [M, K_new]
                exist_pseudo_tensor = exist_pseudo[b][0].cuda()  # [M, K_exist]
                cur_pseudo = torch.cat([new_pseudo, exist_pseudo_tensor], dim=-1)  # [M , K_new + K_exist]
                cur_bcyl = torch.cat([new_bcyl, exist_pseudo_tensor.float()], dim=-1)  # [M , K_new + K_exist]
                cur_score = torch.cat(
                    [torch.cat(obj_score[b]), exist_pseudo[b][1]], dim=-1).cuda()

                # Count new pseudo masks and compute TP
                if num_new_pseudo > 0:
                    self.epoch_new_pseudo_predicted += num_new_pseudo
                    if scene_gt_count > 0:
                        for new_mask_id in range(num_new_pseudo):
                            new_mask = new_pseudo[:, new_mask_id]
                            if new_mask.sum() > 0:
                                max_iou = self.compute_iou_with_gt(new_mask, GT_mask)
                                # Count TP for each IoU threshold
                                for thr in self.sufficiency_iou_thresholds:
                                    if max_iou > thr:
                                        self.epoch_new_pseudo_tp[thr] += 1
            else:
                cur_pseudo = exist_pseudo[b][0].cuda()  # [M , K]
                cur_bcyl = cur_pseudo.float()  # fallback: use predicted mask as region
                cur_score = exist_pseudo[b][1].cuda()

            if (cur_pseudo.sum(0) > 0).sum() > 0:
                valid_pseudo_idx = torch.where(cur_pseudo.sum(0) > 0)[0]
                cur_pseudo, cur_score = cur_pseudo[:,
                                                   valid_pseudo_idx], cur_score[valid_pseudo_idx]
                cur_bcyl = cur_bcyl[:, valid_pseudo_idx]
                nodup_mask = remove_duplications(cur_pseudo, score=cur_score)
                cur_pseudo, cur_score = cur_pseudo[:,
                                                   nodup_mask], cur_score[nodup_mask]
                cur_bcyl = cur_bcyl[:, nodup_mask]

                # Record statistics after deduplication
                num_merged_pseudo = cur_pseudo.shape[1]
                self.epoch_merged_pseudo_predicted += num_merged_pseudo

                # Count merged pseudo masks and compute TP
                if num_merged_pseudo > 0 and scene_gt_count > 0:
                    for merged_mask_id in range(num_merged_pseudo):
                        merged_mask = cur_pseudo[:, merged_mask_id]
                        if merged_mask.sum() > 0:
                            max_iou = self.compute_iou_with_gt(merged_mask, GT_mask)
                            # Count TP for each IoU threshold
                            for thr in self.sufficiency_iou_thresholds:
                                if max_iou > thr:
                                    self.epoch_merged_pseudo_tp[thr] += 1

                if self.cfg.verbose:
                    print('pseudo mask number:', cur_pseudo.shape[1])
                valid_bs.append(b)

                os.makedirs(os.path.join(self.cfg.save_path,
                            'exist_pseudo'), exist_ok=True)
                exist_pseudo_file = os.path.join(
                    self.cfg.save_path, 'exist_pseudo', scene_name[b] + '.pickle')
                with open(exist_pseudo_file, 'wb') as f:
                    pickle.dump(
                        [cur_pseudo.cpu()[inverse_map[b]].bool(), cur_score.cpu()], f)

                if self.cfg.use_sp:
                    sp_pseudo = (scatter_mean(cur_pseudo.float(),
                                 batch_sp[b], dim=0) >= 0.5).float()
                    pseudo_mask_list.append(sp_pseudo)
                else:
                    pseudo_mask_list.append(cur_pseudo)

        # timing
        self.data_time += time.time() - time_cur
        time_cur = time.time()
        # 3. optimize model
        self.model.train(), self.actor.train(), self.critic.train()
        for iter in range(self.batch_iter):
            self.optimizer_actor.zero_grad(), self.optimizer_critic.zero_grad()
            self.optimizer.zero_grad()
            # mask3d_loss
            if self.cfg.use_sp:
                output_train = self.model(in_field, point2segment=batch_sp, raw_coordinates=feature[:, -3:].cuda(
                ), train_on_segments=self.cfg.use_sp, anchor=batch_anchor)
            else:
                output_train = self.model(
                    in_field, raw_coordinates=feature[:, -3:].cuda(), train_on_segments=self.cfg.use_sp)

            # ppo_loss
            ppo_loss, pg_loss, critic_loss, entropy_loss = self.compute_rl_loss(
                pc, output_train, state_index)

            if len(valid_bs) > 0:  # means in some scene, we found object in RL collecting data process
                mask_loss, dice_loss, class_loss = self.compute_seg_loss(
                    output_train, output_train["pred_masks"], pseudo_mask_list, valid_bs)
                seg_loss = 2 * class_loss + 5 * mask_loss + 2 * dice_loss

            else:
                mask_loss = dice_loss = class_loss = torch.tensor(0)
                seg_loss = 2 * class_loss + 5 * mask_loss + 2 * dice_loss

            loss = ppo_loss + seg_loss
            loss.backward()
            self.optimizer_actor.step(), self.optimizer_critic.step()
            self.optimizer.step()

            # info
            self.loss_dict['loss'] += loss.item()
            self.loss_dict['ppo loss'] += ppo_loss.item()
            self.loss_dict['actor loss'] += pg_loss.item()
            self.loss_dict['critic loss'] += critic_loss.item()
            self.loss_dict['ent loss'] += entropy_loss.item()
            self.loss_dict['seg loss'] += seg_loss.item()
            self.loss_dict['mask loss'] += mask_loss.item()
            self.loss_dict['dice loss'] += dice_loss.item()
            self.loss_dict['class loss'] += class_loss.item()

            self.training_iter += 1
            self.step_reward += step_R / step_num
            self.traj_length += step_num / traj_num

            # timing
            self.optimize_time += time.time() - time_cur
            if self.training_iter % self.logging_interval == 0:
                for key, value in self.loss_dict.items():
                    self.loss_dict[key] = self.loss_dict[key] / \
                        self.logging_interval
                self.step_reward /= self.logging_interval
                self.traj_length /= self.logging_interval
                self.logger.info(
                    '{} Epoch: {} [{}/{} ({:.0f}%)]{}, Loss: {:.3f}, ppo: {:.3f}, actor: {:.3f}, critic: {:.3f}, ent: {:.3f}, StepR: {:.2f}, seg: {:.3f}, mask: {:.3f}, '
                    'dice: {:.3f}, class: {:.3f}, evo_discern: {:.3f}, lr: {:.3e}, Traj: {:.2f}, infer time: {:.1f}s, data time: {:.1f}s, optimize time: {:.1f}s, Elapsed time: {:.1f}s ({} iters)'.format(
                        time.strftime(
                            "%Y-%m-%d %H:%M:%S", time.localtime()), epoch, batch_idx, loader_size,
                        100. * batch_idx / loader_size, epoch * loader_size + batch_idx,
                        self.loss_dict['loss'], self.loss_dict['ppo loss'], self.loss_dict['actor loss'],
                        self.loss_dict['critic loss'], self.loss_dict['ent loss'], self.step_reward,
                        self.loss_dict['seg loss'], self.loss_dict['mask loss'], self.loss_dict['dice loss'],
                        self.loss_dict['class loss'], self.loss_dict['evo discern loss'], self.optimizer.param_groups[0]['lr'],
                        self.traj_length, self.infer_time, self.data_time, self.optimize_time, self.infer_time +
                        self.data_time + self.optimize_time,
                        self.logging_interval))
                self.logger.info(
                    '50iou percent: {:.3f}, 25iou percent: {:.3f}, AVG iou: {:.3f}, AVG 50iou: {:.3f}, AVG 25iou: {:.3f})'.format(
                        self.num_ious50 /
                        (self.num_ious + 1e-5), self.num_ious25 /
                        (self.num_ious + 1e-5),
                        self.ious / (self.num_ious + 1e-5), self.ious50 /
                        (self.num_ious50 + 1e-5),
                        self.ious25 / (self.num_ious25 + 1e-5)))
                self.refresh_info()
            time_cur = time.time()
        del batch_anchor

    def compute_seg_loss(self, output, score, pseudo_mask_list, valid_bs):
        matchings = self.matcher(output, pseudo_mask_list, valid_bs)
        aux_matching = []
        if "aux_outputs" in output:
            for i, aux_outputs in enumerate(output["aux_outputs"]):
                tmp_matching = self.matcher(
                    aux_outputs, pseudo_mask_list, valid_bs)
                aux_matching.append(tmp_matching)

        loss_dice, loss_mask, loss_class = 0, 0, 0
        for actual_bs_index, actual_bs in enumerate(valid_bs):
            matched_slot_num = len(matchings[actual_bs_index][0])
            # score more sp
            loss_mask += compute_sigmoid_ce_loss(score[actual_bs][:, matchings[actual_bs_index][0]].t(
            ), pseudo_mask_list[actual_bs_index][:, matchings[actual_bs_index][1].long()].t(), matched_slot_num)
            loss_dice += compute_dice_loss(score[actual_bs][:, matchings[actual_bs_index][0]].t(
            ), pseudo_mask_list[actual_bs_index][:, matchings[actual_bs_index][1].long()].t(), matched_slot_num)

        target_classes = torch.full(output["pred_logits"][valid_bs].shape[:-1],
                                    self.model.num_classes - 1, dtype=torch.int64, device=output["pred_logits"].device)
        for actual_bs_index, actual_bs in enumerate(valid_bs):
            # 0 means chair/forground
            target_classes[actual_bs_index,
                           matchings[actual_bs_index][0].long()] = 0
        loss_class += F.cross_entropy(output["pred_logits"][valid_bs].transpose(
            1, 2), target_classes, ignore_index=-1)

        if "aux_outputs" in output:
            for i, aux_outputs in enumerate(output["aux_outputs"]):
                aux_mask = aux_outputs['pred_masks']
                aux_logits = aux_outputs['pred_logits']
                tmp_matching = aux_matching[i]
                for actual_bs_index, actual_bs in enumerate(valid_bs):
                    tmp_matched_slot_num = len(
                        tmp_matching[actual_bs_index][0])
                    loss_mask += compute_sigmoid_ce_loss(aux_mask[actual_bs][:, tmp_matching[actual_bs_index][0]].t(
                    ), pseudo_mask_list[actual_bs_index][:, tmp_matching[actual_bs_index][1].long()].t(), tmp_matched_slot_num)
                    loss_dice += compute_dice_loss(aux_mask[actual_bs][:, tmp_matching[actual_bs_index][0]].t(
                    ), pseudo_mask_list[actual_bs_index][:, tmp_matching[actual_bs_index][1].long()].t(), tmp_matched_slot_num)

                target_classes = torch.full(
                    aux_logits[valid_bs].shape[:-1], self.model.num_classes - 1, dtype=torch.int64, device=aux_logits.device)
                for actual_bs_index, actual_bs in enumerate(valid_bs):
                    target_classes[actual_bs_index, aux_matching[i]
                                   [actual_bs_index][0].long()] = 0
                loss_class += F.cross_entropy(aux_logits[valid_bs].transpose(
                    1, 2), target_classes, ignore_index=-1)
        return loss_mask, loss_dice, loss_class

    def compute_rl_loss(self, pc, output, state_index):
        device = torch.device("cuda")
        bkb_feature = output["mask_features"]  # .decomposed_features
        #######
        state_info = []
        for b in range(len(pc)):
            state_info.append([])
            for anchor_idx in range(self.cfg.env_num):
                state_info[-1].append(None)

        for traj_id in self.traj_dict.keys():
            env_idx = state_index[self.traj_dict[str(
                traj_id)]['cur_bs']][self.traj_dict[str(traj_id)]['cur_env']]
            tmp_anchor, env_xyz, env_feature = self.traj_dict[traj_id]['initial_bcyl_center'], pc[self.traj_dict[str(
                traj_id)]['cur_bs']][env_idx], bkb_feature[self.traj_dict[str(traj_id)]['cur_bs']][env_idx]
            state_info[self.traj_dict[str(traj_id)]['cur_bs']][self.traj_dict[str(
                traj_id)]['cur_env']] = (env_feature, env_xyz, tmp_anchor)

        if len(self.memory) < self.BATCH_SIZE:
            transitions = self.memory.sample(len(self.memory))
        else:
            transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        action_batch = torch.cat(batch.action).view(-1, 2).long().to(device)
        logprob_batch = torch.tensor(
            batch.logprob, device=device).view(-1, 1)  # .half()
        value_batch = torch.tensor(
            batch.value, device=device).view(-1, 1)  # .half()
        td_target_batch = torch.tensor(
            batch.td_target, device=device).view(-1, 1)  # .half()
        advantage_batch = torch.tensor(
            batch.advantage, device=device).view(-1, 1)  # .half()
        if self.normalize_adv:
            mu, sigma = self.memory.adv_mean_std()
            mu, sigma = torch.tensor(mu).cuda(), torch.tensor(sigma).cuda()
            advantage_batch = (advantage_batch - mu) / (sigma + 1e-8)

        curpos = torch.cat([batch.state[i][0]
                           for i in range(len(batch.state))]).cuda()  # [bs, 3]
        curR = [batch.state[i][1] for i in range(len(batch.state))]
        # history = torch.cat([batch.state[i][2] for i in range(len(batch.state))]).cuda() ##[bs, C]
        bs_idx = torch.tensor([batch.state[i][3] for i in range(
            len(batch.state))]).cuda().long()  # [bs]
        env_idx = torch.tensor([batch.state[i][4] for i in range(
            len(batch.state))]).cuda().long()  # [bs]
        initial_bcyl_center = torch.cat(
            [batch.state[i][5] for i in range(len(batch.state))]).cuda()  # [bs, 3]

        cur_envfeat_list, cur_centered_pos, cur_centered_envxyz = [], [], []
        cur_inbcyl_xyz, cur_inbcyl_feats = [], []
        for idx in range(len(bs_idx)):
            cur_centered_pos.append((curpos[idx].unsqueeze(
                0) - initial_bcyl_center[idx].unsqueeze(0)).unsqueeze(0))
            env_feature, env_xyz, anchor_use2checke = state_info[bs_idx[idx]][env_idx[idx]]
            cur_envfeat_list.append(env_feature), cur_centered_envxyz.append(
                env_xyz - initial_bcyl_center[idx])

            _, bcyl_mask, _ = self.compute_bcyl([], env_xyz, curpos[idx].unsqueeze(
                0), initial_bcyl_center[idx].unsqueeze(0), r=curR[idx])
            inbcyl_xyz, inbcyl_feats = (env_xyz - initial_bcyl_center[idx].unsqueeze(0))[
                torch.where(bcyl_mask)[0]], env_feature[torch.where(bcyl_mask)[0]]

            sample_idx = np.random.choice(inbcyl_xyz.shape[0], self.self_atten_sample_num, replace=False) if inbcyl_xyz.shape[0] >= self.self_atten_sample_num \
                else np.random.choice(inbcyl_xyz.shape[0], self.self_atten_sample_num, replace=True)

            inbcyl_xyz, inbcyl_feats = inbcyl_xyz[sample_idx], inbcyl_feats[sample_idx]
            cur_inbcyl_xyz.append(inbcyl_xyz.unsqueeze(
                0)), cur_inbcyl_feats.append(inbcyl_feats.unsqueeze(0))

        cur_centered_pos = torch.cat(cur_centered_pos)
        cur_centered_pos[:, :, -1] *= 0

        # history_embedding = self.actor.foward_hist(history)
        logits_moving, logits_scale, hidden, state_feats = self.actor(torch.cat(
            cur_inbcyl_xyz), torch.tensor(curR), torch.cat(cur_inbcyl_feats), cur_centered_pos, history=None)

        curr_moving_probs, curr_scale_probs = F.softmax(
            logits_moving, dim=-1), F.softmax(logits_scale, dim=-1)
        curr_value = self.critic(hidden)
        logratio = curr_moving_probs.log().gather(1, action_batch[:, 0].unsqueeze(
            -1)) + curr_scale_probs.log().gather(1, action_batch[:, 1].unsqueeze(-1)) - logprob_batch
        ratio = logratio.exp()

        # Policy loss
        pg_loss1 = advantage_batch * ratio
        pg_loss2 = advantage_batch * \
            torch.clamp(ratio, 1 - self.clip_actor_eps,
                        1 + self.clip_actor_eps)
        pg_loss = -torch.min(pg_loss1, pg_loss2).mean()

        # Critic loss
        if self.clip_value:
            v_loss_unclipped = (curr_value - td_target_batch) ** 2
            v_clipped = value_batch + \
                torch.clamp(curr_value - value_batch, -
                            self.clip_value_eps, self.clip_value_eps)
            v_loss_clipped = (v_clipped - td_target_batch).pow(2)
            critic_loss = 0.5 * \
                torch.max(v_loss_unclipped, v_loss_clipped).mean()
        else:
            critic_loss = 0.5 * \
                (curr_value - td_target_batch.detach()).pow(2).mean()
        ####
        curr_probs = (curr_moving_probs[:, None, :] * curr_scale_probs[:,
                      :, None]).view(curr_moving_probs.shape[0], -1)
        entropy_loss = - (curr_probs * curr_probs.log()).sum(-1).mean()

        ppo_loss = pg_loss + 1 * critic_loss - self.ent_coeff * entropy_loss
        return ppo_loss, pg_loss, critic_loss, entropy_loss

    def center_points(self, points):
        """
        Center point cloud at origin without scaling.
        Only translation, no scale normalization.
        """
        centroid = points.mean(dim=0)
        centered_points = points - centroid
        return centered_points

    def prepare_point_transformer_input(self, points, masks=None, device='cuda'):
        """
        Prepare input data for Point Transformer V2 model

        Args:
            points: [batch_size, num_points, 3] torch tensor (variable num_points allowed)
            masks: [batch_size, num_points] torch tensor or None
            device: device to place tensors on

        Returns:
            dict containing:
                - coord: [total_points, 3] point coordinates (centered)
                - feat: [total_points, 3] features (centered point coordinates)
                - offset: [batch_size] cumulative point counts
                - labels: [total_points] point labels (if masks provided)
                - original_points: original points tensor
                - original_masks: original masks tensor (if provided)
        """
        batch_size = len(points)
        all_coords = []
        all_features = []
        all_labels = []
        batch_offsets = [0]

        for b in range(batch_size):
            coords = points[b].to(device).float()
            coords_centered = coords - coords.mean(dim=0, keepdim=True)
            features = coords_centered
            all_coords.append(coords_centered)
            all_features.append(features)
            batch_offsets.append(batch_offsets[-1] + coords.shape[0])
            if masks is not None:
                labels = masks[b].to(device).long()
                all_labels.append(labels)

        coord = torch.cat(all_coords, dim=0)
        feat = torch.cat(all_features, dim=0)
        offset = torch.tensor(
            batch_offsets[1:], dtype=torch.long, device=device)

        out = {
            'coord': coord,
            'feat': feat,
            'offset': offset,
            'original_points': points
        }
        if masks is not None:
            labels = torch.cat(all_labels, dim=0)
            out['labels'] = labels
            out['original_masks'] = masks
        return out

    def compute_sdfmask(self, traj_id_list, iters=1, max_iters=1, batch_sp=None, state_index=None):
        if self.discern_backend == "sparseUnet":
            return self.compute_sdfmask_sparseUnet(traj_id_list, iters, max_iters, batch_sp, state_index)
        if self.discern_backend == "pointTransformer":
            return self.compute_sdfmask_pointTransformer(traj_id_list, iters, max_iters, batch_sp, state_index)
        if self.discern_backend == "pointNet":
            return self.compute_sdfmask_pointNet(traj_id_list, iters, max_iters, batch_sp, state_index)
        raise ValueError(f"Unknown discern_backend: {self.discern_backend!r}")

    def compute_sdfmask_sparseUnet(self, traj_id_list, iters=1, max_iters=1, batch_sp=None, state_index=None):
        # Initialize masks and valid trajectory list
        final_mask = {}
        valid_traj_id = []
        for traj_id in traj_id_list:
            env_xyz = self.traj_dict[traj_id]['env_xyz']
            pre_center = self.traj_dict[traj_id]['curpos']
            pre_center[:, -1] = self.obj_center_z
            mask1 = self.compute_cur_ball(env_xyz, pre_center, r=0.6)[1]
            mask2 = self.compute_cur_ball(
                env_xyz, pre_center, r=self.traj_dict[traj_id]['curR'])[1]
            if mask2.sum() > self.mask_min_size and mask1.sum() > self.bcyl_min_size:
                tmp_pc = env_xyz[torch.where(
                    self.traj_dict[traj_id]['bcyl_mask'])[0]]
                size = (tmp_pc - tmp_pc.mean(0, keepdim=True)
                        )[:, :2].norm(p=2, dim=-1).max()
                if size >= self.min_obj_r and size <= self.max_obj_r:
                    valid_traj_id.append(traj_id)
            final_mask[traj_id] = torch.zeros_like(env_xyz)[:, 0]

        # If no valid trajectories, return directly
        if not valid_traj_id:
            return list(final_mask.values()), None, None, None


        voxel_data_list = []
        for traj_id in valid_traj_id:
            env_xyz = self.traj_dict[traj_id]['env_xyz']
            bcyl_mask = self.compute_cur_bcyl(
                env_xyz, self.traj_dict[traj_id]['curpos'],
                r=self.traj_dict[traj_id]['curR'])[1]
            bcyl_idxs = torch.where(bcyl_mask)[0]
            bcyl_pc = env_xyz[bcyl_idxs]  

            # Create sparse tensor data for this trajectory
            voxel_data = {
                'points': bcyl_pc, 
                'bcyl_idxs': bcyl_idxs,
                'num_points': bcyl_pc.shape[0]
            }
            voxel_data_list.append(voxel_data)

        # Batch process all valid trajectories using SpConv voxelization
        if len(voxel_data_list) > 0:
            voxel_size = 0.05  
            scale = 1.0 / voxel_size
            device = voxel_data_list[0]['points'].device

            all_grid_coords = []
            all_features = []
            batch_offsets = [0]
            batch_results = []

            for i, vd in enumerate(voxel_data_list):
                points = vd['points']  # [N_i, 3]
                points = points - points.mean(dim=0, keepdim=True)  # center
                coords_np = points.cpu().numpy()  # [num_points, 3]

                # 1. Subtract minimum to ensure non-negative coordinates
                grid_coords = coords_np - coords_np.min(0)
                grid_coords = np.floor(grid_coords * scale)
                grid_coords_unique, unq_idx, unq_inv = np.unique(
                    grid_coords, return_index=True, return_inverse=True, axis=0)

                grid_coords_unique = torch.from_numpy(
                    grid_coords_unique).long().to(device)
                unq_idx = torch.from_numpy(unq_idx).long()
                unq_inv = torch.from_numpy(unq_inv).long().to(device)
                unq_features = points[unq_idx].to(device)

                all_grid_coords.append(grid_coords_unique)
                all_features.append(unq_features)
                batch_offsets.append(
                    batch_offsets[-1] + grid_coords_unique.shape[0])
                batch_results.append({
                    'unq_idx': unq_idx,
                    'unq_inv': unq_inv,
                    'num_unique': grid_coords_unique.shape[0]
                })

            # Concatenate all batches
            grid_coord = torch.cat(all_grid_coords, dim=0)
            feat = torch.cat(all_features, dim=0)
            offset = torch.tensor(
                batch_offsets[1:], dtype=torch.long, device=device)

            # Create data dict for SpConv network
            data_dict = {
                'grid_coord': grid_coord,
                'feat': feat,
                'offset': offset
            }
            self.discern_net.eval()
            # Forward pass through SpConv network
            with torch.no_grad():
                logits = self.discern_net(data_dict)  
                probs = torch.softmax(logits, dim=-1) 

            # Split results back to individual trajectories
            start_idx = 0
            for i, traj_id in enumerate(valid_traj_id):
                end_idx = start_idx + batch_results[i]['num_unique']
                unique_probs = probs[start_idx:end_idx]
                unq_inv = batch_results[i]['unq_inv']
                point_probs = unique_probs[unq_inv]  
                fg_probs = point_probs[:, 1]  
                bg_probs = point_probs[:, 0]  
                hard_fg = fg_probs > bg_probs
                bcyl_idxs = voxel_data_list[i]['bcyl_idxs']

                # Update final mask
                final_mask[traj_id][bcyl_idxs] = hard_fg.float()
                W = torch.zeros_like(self.traj_dict[traj_id]['W'])
                W[bcyl_idxs] = fg_probs
                self.traj_dict[traj_id]['W'] = W

                # Mark as invalid if foreground points are insufficient or height exceeds limit
                if (W > 0.1).sum() < self.mask_min_size or \
                        self.traj_dict[traj_id]['env_xyz'][torch.where(W != 0)].max(0).values[-1] > self.max_obj_h:
                    final_mask[traj_id] = torch.zeros_like(final_mask[traj_id])

                # batch_sp pseudo label projection
                if batch_sp is not None:
                    tmp_bs = self.traj_dict[traj_id]['cur_bs']
                    tmp_env = self.traj_dict[traj_id]['cur_env']
                    env_batch_sp = batch_sp[tmp_bs][state_index[tmp_bs][tmp_env]]
                    valid_idx = torch.where(env_batch_sp != -1)[0].long()
                    sp_pseudo = (
                        scatter_mean(
                            final_mask[traj_id].float()[valid_idx],
                            env_batch_sp[valid_idx], dim=0
                        ) >= 0.5
                    ).float()
                    final_mask[traj_id][valid_idx] = sp_pseudo[env_batch_sp[valid_idx]]

                start_idx = end_idx

        # Collect output
        final_inmask_prob, final_inmask_pc, final_inmask_norm = [], [], []
        for traj_id in final_mask:
            idxs = torch.where(final_mask[traj_id])[0]
            final_inmask_prob.append(self.traj_dict[traj_id]['W'][idxs])
            final_inmask_pc.append(self.traj_dict[traj_id]['env_xyz'][idxs])
            final_inmask_norm.append(self.traj_dict[traj_id]['env_norm'][idxs])

        return list(final_mask.values()), final_inmask_pc, final_inmask_norm, final_inmask_prob

    def compute_sdfmask_pointTransformer(self, traj_id_list, iters=1, max_iters=1, batch_sp=None, state_index=None):
        final_mask = {}
        valid_traj_id = []
        for traj_id in traj_id_list:
            env_xyz = self.traj_dict[traj_id]['env_xyz']
            pre_center = self.traj_dict[traj_id]['curpos']
            pre_center[:, -1] = self.obj_center_z
            mask1 = self.compute_cur_ball(env_xyz, pre_center, r=0.6)[1]
            mask2 = self.compute_cur_ball(
                env_xyz, pre_center, r=self.traj_dict[traj_id]['curR'])[1]
            if mask2.sum() > self.mask_min_size and mask1.sum() > self.bcyl_min_size:
                tmp_pc = env_xyz[torch.where(
                    self.traj_dict[traj_id]['bcyl_mask'])[0]]
                size = (tmp_pc - tmp_pc.mean(0, keepdim=True)
                        )[:, :2].norm(p=2, dim=-1).max()
                if size >= self.min_obj_r and size <= self.max_obj_r:
                    valid_traj_id.append(traj_id)
            final_mask[traj_id] = torch.zeros_like(env_xyz)[:, 0]

        # If no valid trajectories, return directly
        if not valid_traj_id:
            return list(final_mask.values()), None, None, None

        pc_list, idxs_list = [], []
        orig_M_list = []

        for traj_id in valid_traj_id:
            env_xyz = self.traj_dict[traj_id]['env_xyz']
            bcyl_mask = self.compute_cur_bcyl(
                env_xyz, self.traj_dict[traj_id]['curpos'],
                r=self.traj_dict[traj_id]['curR'])[1]
            bcyl_idxs = torch.where(bcyl_mask)[0]
            bcyl_pc = env_xyz[bcyl_idxs]
            pc_list.append(bcyl_pc)
            idxs_list.append(bcyl_idxs)
            orig_M_list.append(bcyl_pc.shape[0])

        # Batch forward with Point Transformer V2
        point_data = self.prepare_point_transformer_input(
            pc_list, masks=None, device='cuda')
        data_dict = {
            'coord': point_data['coord'],
            'feat': point_data['feat'],
            'offset': point_data['offset']
        }
        # Use eval mode for inference (compute_sdfmask is inference, not training)
        self.discern_net.eval()
        with torch.no_grad():
            logits = self.discern_net(data_dict)
            probs = F.softmax(logits, dim=1)

        for i, traj_id in enumerate(valid_traj_id):
            M_orig = orig_M_list[i]
            bcyl_idxs = idxs_list[i]
            start_idx = point_data['offset'][i - 1].item() if i > 0 else 0
            end_idx = point_data['offset'][i].item()
            probs_i = probs[start_idx:end_idx]
            probs_i = probs_i[:M_orig]
            fg_all = probs_i[:, 1]
            bg_all = probs_i[:, 0]
            hard_fg = fg_all > bg_all

            final_mask[traj_id][bcyl_idxs] = hard_fg.float()
            W = torch.zeros_like(self.traj_dict[traj_id]['W'])
            W[bcyl_idxs] = fg_all
            self.traj_dict[traj_id]['W'] = W

            # Mark as invalid if foreground points are insufficient or height exceeds limit
            if (W > 0.1).sum() < self.mask_min_size or \
                    self.traj_dict[traj_id]['env_xyz'][torch.where(W != 0)].max(0).values[-1] > self.max_obj_h:
                final_mask[traj_id] = torch.zeros_like(final_mask[traj_id])

            # batch_sp pseudo label projection
            if batch_sp is not None:
                tmp_bs = self.traj_dict[traj_id]['cur_bs']
                tmp_env = self.traj_dict[traj_id]['cur_env']
                env_batch_sp = batch_sp[tmp_bs][state_index[tmp_bs][tmp_env]]
                valid_idx = torch.where(env_batch_sp != -1)[0].long()
                sp_pseudo = (
                    scatter_mean(
                        final_mask[traj_id].float()[valid_idx],
                        env_batch_sp[valid_idx], dim=0
                    ) >= 0.5
                ).float()
                final_mask[traj_id][valid_idx] = sp_pseudo[env_batch_sp[valid_idx]]

        # collect ouput
        final_inmask_prob, final_inmask_pc, final_inmask_norm = [], [], []
        for traj_id in final_mask:
            idxs = torch.where(final_mask[traj_id])[0]
            final_inmask_prob.append(self.traj_dict[traj_id]['W'][idxs])
            final_inmask_pc.append(self.traj_dict[traj_id]['env_xyz'][idxs])
            final_inmask_norm.append(self.traj_dict[traj_id]['env_norm'][idxs])

        return list(final_mask.values()), final_inmask_pc, final_inmask_norm, final_inmask_prob

    def compute_sdfmask_pointNet(self, traj_id_list, iters=1, max_iters=1, batch_sp=None, state_index=None):
        # Initialize masks and valid trajectory list
        final_mask = {}
        valid_traj_id = []
        for traj_id in traj_id_list:
            env_xyz = self.traj_dict[traj_id]['env_xyz']
            pre_center = self.traj_dict[traj_id]['curpos']
            pre_center[:, -1] = self.obj_center_z
            mask1 = self.compute_cur_ball(env_xyz, pre_center, r=0.6)[1]
            mask2 = self.compute_cur_ball(
                env_xyz, pre_center, r=self.traj_dict[traj_id]['curR'])[1]
            if mask2.sum() > self.mask_min_size and mask1.sum() > self.bcyl_min_size:
                tmp_pc = env_xyz[torch.where(
                    self.traj_dict[traj_id]['bcyl_mask'])[0]]
                size = (tmp_pc - tmp_pc.mean(0, keepdim=True)
                        )[:, :2].norm(p=2, dim=-1).max()
                if size >= self.min_obj_r and size <= self.max_obj_r:
                    valid_traj_id.append(traj_id)
            final_mask[traj_id] = torch.zeros_like(env_xyz)[:, 0]

        # If no valid trajectories, return directly
        if not valid_traj_id:
            return list(final_mask.values()), None, None, None

        # pad to 4096 for batch process acceleration
        PAD_N = 4096
        pc_list, idxs_list = [], []
        orig_M_list, pad_M_list = [], []
        for traj_id in valid_traj_id:
            env_xyz = self.traj_dict[traj_id]['env_xyz']
            bcyl_mask = self.compute_cur_bcyl(
                env_xyz, self.traj_dict[traj_id]['curpos'],
                r=self.traj_dict[traj_id]['curR'])[1]
            bcyl_idxs = torch.where(bcyl_mask)[0]
            bcyl_pc = env_xyz[bcyl_idxs] - \
                env_xyz[bcyl_idxs].mean(0, keepdim=True)
            M = bcyl_pc.shape[0]
            # padding to fix number of point
            if M < PAD_N:
                extra = torch.randint(0, M, (PAD_N - M,),
                                      device=bcyl_pc.device)
                pc_input = torch.cat([bcyl_pc, bcyl_pc[extra]], dim=0)
                M_pad = PAD_N
            else:
                pc_input = bcyl_pc
                M_pad = M

            pc_list.append(pc_input)
            idxs_list.append(bcyl_idxs)
            orig_M_list.append(M)
            pad_M_list.append(M_pad)

        # batched input
        feats_batch = torch.stack([
            pc.transpose(0, 1)  # [3, M]
            for pc in pc_list
        ], dim=0).cuda()  # Result shape [B, 3, M]

        # batch forward
        self.discern_net.eval()
        with torch.no_grad():
            logp_batch, _ = self.discern_net(feats_batch)
            probs_batch = logp_batch.exp()  # [B, M_max, 2]

        # Split batch outputs, write back final_mask & traj_dict, apply per-sample post-processing
        for i, traj_id in enumerate(valid_traj_id):
            M_orig = orig_M_list[i]
            bcyl_idxs = idxs_list[i]
            probs = probs_batch[i]
            fg_all = probs[:M_orig, 1]
            bg_all = probs[:M_orig, 0]
            hard_fg = fg_all > bg_all

            final_mask[traj_id][bcyl_idxs] = hard_fg.float()
            W = torch.zeros_like(self.traj_dict[traj_id]['W'])
            W[bcyl_idxs] = fg_all
            self.traj_dict[traj_id]['W'] = W

            # Mark as invalid if foreground points are insufficient or height exceeds limit
            if (W > 0.1).sum() < self.mask_min_size or \
                    self.traj_dict[traj_id]['env_xyz'][torch.where(W != 0)].max(0).values[-1] > self.max_obj_h:
                final_mask[traj_id] = torch.zeros_like(final_mask[traj_id])

            # batch_sp pseudo label projection
            if batch_sp is not None:
                tmp_bs = self.traj_dict[traj_id]['cur_bs']
                tmp_env = self.traj_dict[traj_id]['cur_env']
                env_batch_sp = batch_sp[tmp_bs][state_index[tmp_bs][tmp_env]]
                valid_idx = torch.where(env_batch_sp != -1)[0].long()
                sp_pseudo = (
                    scatter_mean(
                        final_mask[traj_id].float()[valid_idx],
                        env_batch_sp[valid_idx], dim=0
                    ) >= 0.5
                ).float()
                final_mask[traj_id][valid_idx] = sp_pseudo[env_batch_sp[valid_idx]]

        # collect ouput
        final_inmask_prob, final_inmask_pc, final_inmask_norm = [], [], []
        for traj_id in final_mask:
            idxs = torch.where(final_mask[traj_id])[0]
            final_inmask_prob.append(self.traj_dict[traj_id]['W'][idxs])
            final_inmask_pc.append(self.traj_dict[traj_id]['env_xyz'][idxs])
            final_inmask_norm.append(self.traj_dict[traj_id]['env_norm'][idxs])

        return list(final_mask.values()), final_inmask_pc, final_inmask_norm, final_inmask_prob

    def compute_convergence(self, mask_list, pc_list, prob_list, vis_comp: bool = False, save_dir: str = "./vis_comp"):
        # some mask are all zeros
        valid_mask = [(mask.sum() >= self.mask_min_size).item()
                      for mask in mask_list]
        comp_input = []
        for item_idx, (pc, prob, validness) in enumerate(zip(pc_list, prob_list, valid_mask.copy())):
            if validness:
                # Always sample 1024 points for comp input
                idx = torch.multinomial(
                    prob, num_samples=1024, replacement=True)
                sampled_pc = pc[idx]
                center = sampled_pc.mean(0, keepdim=True)
                mask_size = (pc - center)[:, 0:2].norm(p=2, dim=-1).max()
                mask_height = pc[:, -1].max()
                if mask_size >= self.min_obj_r and mask_size <= self.max_obj_r and mask_height <= self.max_obj_h:
                    bbox_min, bbox_max = (
                        sampled_pc - center).min(0).values, (sampled_pc - center).max(0).values
                    scale = (bbox_max - bbox_min).max() + 1e-6
                    comp_input.append(
                        ((sampled_pc - center).unsqueeze(0)) / scale)
                else:
                    valid_mask[item_idx] = False

        pc2mesh_list, mesh2pc_list, sec_validness = [], [], []
        if np.array(valid_mask).sum() > 0:
            comp_input = torch.cat(comp_input)  
            comp_centers = comp_input.mean(dim=1, keepdim=True)  
            comp_input = comp_input - comp_centers 
            dists = torch.norm(comp_input, dim=2)  
            comp_scale = dists.max(dim=1, keepdim=True).values 
            comp_scale = comp_scale + 1e-6 
            comp_input = comp_input / comp_scale.unsqueeze(-1) 

            with torch.no_grad():
                out = self.comp_net(comp_input.cuda())
            comp_output = out[-1] 

            if vis_comp and comp_output.shape[0] > 0:
                os.makedirs(save_dir, exist_ok=True)
                B = comp_output.shape[0]
                comp_out_cpu = comp_output.detach().cpu()
                for b in range(B):
                    pts_in = comp_input[b].detach().cpu()
                    pts_out = comp_out_cpu[b]
                    pts = torch.cat([pts_in, pts_out], dim=0)
                    n_in = pts_in.shape[0]
                    # use different color to seprate input and output of completion model
                    mask = torch.zeros(pts.shape[0], dtype=torch.uint8)
                    mask[n_in:] = 1
                    save_ply_with_mask(
                        pts,
                        mask,
                        os.path.join(save_dir, f"comp_net_io_{b:03d}.ply"),
                    )
                print(f"[comp_net] saved {B} PLY (gray=input, red=output, same normalized space) to {save_dir}")

            # Denormalize, inorder to send it into objnet
            comp_output = comp_output * comp_scale.unsqueeze(-1) + comp_centers  # [B, N, 3]

            B, N, _ = comp_output.shape
            if N != 1024:
                if N > 1024:
                    idx = torch.randperm(N)[:1024]
                    comp_output = comp_output[:, idx, :]
                else:
                    idx = torch.randint(
                        0, N, (B, 1024), device=comp_output.device)
                    comp_output = torch.stack(
                        [comp_output[b][idx[b]] for b in range(B)], dim=0)

            head = 0
            with torch.no_grad():
                while head < comp_output.shape[0]:
                    embedd = self.objnet.encode(
                        comp_output[head:min(head + self.sdf_bs, comp_output.shape[0])])

                    canonical_query_pc = comp_output[head:min(head + self.sdf_bs,
                                                              comp_output.shape[0])]  
                    mesh_pts, validness = self.extract_shape_pts(self.objnet.decode, embedd,
                                                                 sample_pts_num=comp_output.shape[1], N=32)

                    batch_pc2mesh = 1000 * \
                        torch.ones_like(canonical_query_pc)[:, :, 1]
                    batch_mesh2pc = 1000 * \
                        torch.ones_like(canonical_query_pc)[:, :, 1]
                    if len(mesh_pts) > 0:
                        valid_canonical_query_pc = canonical_query_pc[np.where(validness)[
                            0]]

                        pc2mesh = (mesh_pts[:, :, None, :] - valid_canonical_query_pc[:, None, :, :]).norm(p=2,
                                                                                                           dim=-1).min(
                            1).values
                        mesh2pc = (mesh_pts[:, :, None, :] - valid_canonical_query_pc[:, None, :, :]).norm(p=2,
                                                                                                           dim=-1).min(
                            2).values

                        batch_pc2mesh[np.where(validness)[0]] = pc2mesh
                        batch_mesh2pc[np.where(validness)[0]] = mesh2pc
                        # print(f'pc2mesh: {pc2mesh} mesh2pc: {mesh2pc}')
                    pc2mesh_list.append(batch_pc2mesh)  # shape: [bs,1024]
                    mesh2pc_list.append(batch_mesh2pc)
                    head += self.sdf_bs

            # use numpy to update list
            valid_mask_array = np.array(valid_mask)
            valid_mask = valid_mask_array.tolist()
            if len(pc2mesh_list) > 0:
                return valid_mask, torch.cat(pc2mesh_list), torch.cat(mesh2pc_list)
            else:
                return valid_mask, [], []
        else:
            return valid_mask, [], []

    def select_action(self, xyz, R, feats, curpos, history):
        with torch.no_grad():
            logits_moving, logits_scale, hidden, state_feats = self.actor(
                xyz, R, feats, curpos,  history=None)
            prob_moving, prob_scale = F.softmax(
                logits_moving, dim=-1).detach(), F.softmax(logits_scale, dim=-1).detach()
            value = self.critic(hidden).detach()
            moving_action_dist, scale_action_dist = Categorical(
                probs=prob_moving.cpu()), Categorical(probs=prob_scale.cpu())
            moving_action, scale_action = moving_action_dist.sample(), scale_action_dist.sample()
            return torch.cat((moving_action.unsqueeze(-1), scale_action.unsqueeze(-1)), dim=-1), moving_action_dist.log_prob(moving_action) + scale_action_dist.log_prob(scale_action), value, state_feats

    def select_best_action(self, xyz, R, feats, curpos, history):
        with torch.no_grad():
            logits_moving, logits_scale, hidden, state_feats = self.actor(
                xyz, R, feats, curpos, history=None
            )
            prob_moving, prob_scale = F.softmax(logits_moving, dim=-1).detach(), F.softmax(
                logits_scale, dim=-1
            ).detach()
            moving_action, scale_action = torch.max(prob_moving, 1).indices, torch.max(prob_scale, 1).indices
            return torch.cat((moving_action.unsqueeze(-1), scale_action.unsqueeze(-1)), dim=-1), state_feats

    def compute_cur_bcyl(self, env_xyz, initial_bcyl_center, r=None):
        if r is None:
            r = self.initial_R
        bcyl_center = initial_bcyl_center.clone()
        bcyl_mask = torch.logical_and(
            (env_xyz - bcyl_center)[:, 0:2].norm(p=2, dim=-1) <= r, env_xyz[:, -1] <= self.obj_h)
        return bcyl_center, bcyl_mask, r

    def compute_cur_ball(self, env_xyz, initial_bcyl_center, r=None):
        if r is None:
            r = self.initial_R
        bcyl_center = initial_bcyl_center.clone()
        bcyl_mask = torch.logical_and(
            (env_xyz - bcyl_center).norm(p=2, dim=-1) <= r, env_xyz[:, -1] <= self.obj_h)
        return bcyl_center, bcyl_mask, r

    def compute_bcyl(self, action, env_xyz, cur_cyl_loc, initial_cyl_cur, r=None):
        if r is None:
            r = self.initial_R
        if len(action) > 0:
            moving_action, scale_action = action[:, 0], action[:, 1]
            # moving
            if moving_action == 1:
                tmp_bcyl_center = cur_cyl_loc.clone()
                tmp_bcyl_center[:, 0] = cur_cyl_loc[:, 0] + self.moving_step
                if (tmp_bcyl_center - initial_cyl_cur)[:, 0:2].norm(p=2, dim=-1) > self.anchor_env_r:
                    tmp_bcyl_center[:, 0] = cur_cyl_loc[:, 0] + torch.sqrt(
                        self.anchor_env_r**2 - (tmp_bcyl_center-initial_cyl_cur)[:, 1]**2)
                tmp_bcyl_mask = torch.logical_and(
                    (env_xyz - tmp_bcyl_center)[:, 0:2].norm(p=2, dim=-1) <= r, env_xyz[:, -1] <= self.obj_h)
                if tmp_bcyl_mask.sum() > self.bcyl_min_size:
                    cur_cyl_loc = tmp_bcyl_center

            elif moving_action == 2:
                tmp_bcyl_center = cur_cyl_loc.clone()
                tmp_bcyl_center[:, 0] = cur_cyl_loc[:, 0] - self.moving_step
                if (tmp_bcyl_center - initial_cyl_cur)[:, 0:2].norm(p=2, dim=-1) > self.anchor_env_r:
                    tmp_bcyl_center[:, 0] = cur_cyl_loc[:, 0] - torch.sqrt(
                        self.anchor_env_r**2 - (tmp_bcyl_center-initial_cyl_cur)[:, 1]**2)
                tmp_bcyl_mask = torch.logical_and(
                    (env_xyz - tmp_bcyl_center)[:, 0:2].norm(p=2, dim=-1) <= r, env_xyz[:, -1] <= self.obj_h)
                if tmp_bcyl_mask.sum() > self.bcyl_min_size:
                    cur_cyl_loc = tmp_bcyl_center

            elif moving_action == 3:
                tmp_bcyl_center = cur_cyl_loc.clone()
                tmp_bcyl_center[:, 1] = cur_cyl_loc[:, 1] + self.moving_step
                if (tmp_bcyl_center - initial_cyl_cur)[:, 0:2].norm(p=2, dim=-1) > self.anchor_env_r:
                    tmp_bcyl_center[:, 1] = cur_cyl_loc[:, 1] + torch.sqrt(
                        self.anchor_env_r**2 - (tmp_bcyl_center-initial_cyl_cur)[:, 0]**2)
                tmp_bcyl_mask = torch.logical_and(
                    (env_xyz - tmp_bcyl_center)[:, 0:2].norm(p=2, dim=-1) <= r, env_xyz[:, -1] <= self.obj_h)
                if tmp_bcyl_mask.sum() > self.bcyl_min_size:
                    cur_cyl_loc = tmp_bcyl_center

            elif moving_action == 4:
                tmp_bcyl_center = cur_cyl_loc.clone()
                tmp_bcyl_center[:, 1] = cur_cyl_loc[:, 1] - self.moving_step
                if (tmp_bcyl_center - initial_cyl_cur)[:, 0:2].norm(p=2, dim=-1) > self.anchor_env_r:
                    tmp_bcyl_center[:, 1] = cur_cyl_loc[:, 1] - torch.sqrt(
                        self.anchor_env_r**2 - (tmp_bcyl_center-initial_cyl_cur)[:, 0]**2)
                tmp_bcyl_mask = torch.logical_and(
                    (env_xyz - tmp_bcyl_center)[:, 0:2].norm(p=2, dim=-1) <= r, env_xyz[:, -1] <= self.obj_h)
                if tmp_bcyl_mask.sum() > self.bcyl_min_size:
                    cur_cyl_loc = tmp_bcyl_center

            # scaling
            if scale_action == 1:
                if r*self.R_decay >= self.min_R:
                    tmp_r = r*self.R_decay
                    tmp_bcyl_mask = torch.logical_and(
                        (env_xyz - cur_cyl_loc)[:, 0:2].norm(p=2, dim=-1) <= tmp_r, env_xyz[:, -1] <= self.obj_h)
                    if tmp_bcyl_mask.sum() > self.bcyl_min_size:
                        r = tmp_r

            elif scale_action == 2:
                if r/self.R_decay <= self.max_R:
                    tmp_r = r/self.R_decay
                    tmp_bcyl_mask = torch.logical_and(
                        (env_xyz - cur_cyl_loc)[:, 0:2].norm(p=2, dim=-1) <= tmp_r, env_xyz[:, -1] <= self.obj_h)
                    if tmp_bcyl_mask.sum() > self.bcyl_min_size:
                        r = tmp_r

        bcyl_mask = torch.logical_and(
            (env_xyz - cur_cyl_loc)[:, 0:2].norm(p=2, dim=-1) <= r, env_xyz[:, -1] <= self.obj_h)
        return cur_cyl_loc, bcyl_mask, r

    def compute_reward_CD(self, idx, valid_mask, pc2meshs, mesh2pcs, traj_id=None, sdfmask=None):
        if valid_mask[idx]:
            true_list = [i for i, x in enumerate(valid_mask) if x]
            idx_in_true_list = true_list.index(idx)
            pc2mesh = pc2meshs[idx_in_true_list]
            mesh2pc = mesh2pcs[idx_in_true_list]
            cd_distance = pc2mesh.mean() + mesh2pc.mean()

            # Collect data for discerning module evolution
            if (cd_distance <= self.evo_thr and traj_id is not None and sdfmask is not None and
                    self.current_epoch in self.evo_epochs):
                self.collect_evo_training_data(traj_id, sdfmask, cd_distance)

            if cd_distance <= self.cd_thr:
                return self.nu, pc2mesh.mean(), mesh2pc.mean()
            else:
                return -1, pc2mesh.mean(), mesh2pc.mean()
        else:
            return -1, None, None

    def collect_evo_training_data(self, traj_id, sdfmask, cd_distance):
        env_xyz = self.traj_dict[traj_id]['env_xyz']
        bcyl_mask = self.traj_dict[traj_id]['bcyl_mask']
        bcyl_idxs = torch.where(bcyl_mask)[0]
        bcyl_pc = env_xyz[bcyl_idxs]
        sdfmask = sdfmask[bcyl_idxs]
        bcyl_pc_centered = self.center_points(bcyl_pc)
        height_range = bcyl_pc_centered[:,2].max() - bcyl_pc_centered[:, 2].min()
        if height_range > 1.2:
            return
        # save traingning data to ply
        evo_train_dir = os.path.join(
            self.save_path, 'evo_data', f'epoch_{self.current_epoch:03d}')
        os.makedirs(evo_train_dir, exist_ok=True)
        cd_int = int(round(cd_distance.item() * 100))
        train_filepath = os.path.join(
            evo_train_dir, f"{self.evo_sample_counter:06d}_cd{cd_int:02d}_input_with_labels.ply")
        save_ply_with_mask(bcyl_pc_centered, sdfmask, train_filepath)

        # Update sample counter for visualization
        self.evo_sample_counter += 1
        print(
            f"[EVO] (epoch: {self.current_epoch}) Collected evo training data sample {self.evo_sample_counter}")

    def voxelize_batch(self, points, masks, voxel_size=0.05, device='cuda'):
        """
        Voxelize a batch of point clouds using min-subtraction for grid coordinates,
        and use centered coordinates as features.
        """
        batch_size = points.shape[0]
        scale = 1.0 / voxel_size
        all_grid_coords = []
        all_features = []
        all_labels = []
        all_inverse_maps = []
        batch_offsets = [0]

        for b in range(batch_size):
            pts = points[b]  # [N, 3]
            msk = masks[b]   # [N]
            # Center points for features
            centered_pts = pts - pts.mean(dim=0, keepdim=True)
            # Subtract min for grid coordinates
            pts_np = pts.cpu().numpy()
            grid_coords = pts_np - pts_np.min(0)
            grid_coords = np.floor(grid_coords * scale)
            # Unique voxelization
            grid_coords_unique, unq_idx, unq_inv = np.unique(
                grid_coords, return_index=True, return_inverse=True, axis=0)
            # Convert to tensors
            grid_coords_unique = torch.from_numpy(
                grid_coords_unique).long().to(device)
            unq_idx = torch.from_numpy(unq_idx).long()
            unq_inv = torch.from_numpy(unq_inv).long().to(device)
            # Adjust inverse mapping to global voxel indices
            unq_inv_global = unq_inv + batch_offsets[-1]
            all_inverse_maps.append(unq_inv_global)
            # Get corresponding labels and centered features
            unq_labels = msk[unq_idx].to(device)
            unq_features = centered_pts[unq_idx].to(device)
            all_grid_coords.append(grid_coords_unique)
            all_features.append(unq_features)
            all_labels.append(unq_labels)
            batch_offsets.append(
                batch_offsets[-1] + grid_coords_unique.shape[0])

        # Concatenate all batches
        grid_coord = torch.cat(all_grid_coords, dim=0)
        feat = torch.cat(all_features, dim=0)
        labels = torch.cat(all_labels, dim=0)
        offset = torch.tensor(
            batch_offsets[1:], dtype=torch.long, device=device)

        return {
            'grid_coord': grid_coord,
            'feat': feat,
            'offset': offset,
            'labels': labels,
            'inverse_map': all_inverse_maps,
            'original_points': points,
            'original_masks': masks
        }

    def calculate_original_point_accuracy(self, voxel_pred_logits, voxel_data):
        """
        Calculate accuracy on original points by mapping voxel predictions back to points
        """
        # Get voxel predictions
        voxel_predictions = torch.argmax(
            voxel_pred_logits, dim=1)  # [total_voxels]

        # Map voxel predictions back to original points
        point_predictions = []
        point_labels = []
        total_correct = 0
        total_points = 0

        for b, inverse_map in enumerate(voxel_data['inverse_map']):
            batch_point_preds = voxel_predictions[inverse_map]
            batch_point_labels = voxel_data['original_masks'][b].to(
                voxel_pred_logits.device)  # [num_points_in_batch]

            point_predictions.append(batch_point_preds)
            point_labels.append(batch_point_labels)

            # Calculate accuracy for this batch
            correct = (batch_point_preds == batch_point_labels).sum().item()
            total_correct += correct
            total_points += batch_point_labels.size(0)

        accuracy = total_correct / total_points if total_points > 0 else 0.0

        return accuracy, point_predictions, point_labels, total_correct, total_points

    def compute_iou_with_gt(self, pred_mask, gt_mask):
        """
        Compute IoU between a predicted mask and all GT masks.
        Returns the maximum IoU.
        Vectorized version for better performance.

        Args:
            pred_mask: [N] binary mask
            gt_mask: [K, N] binary masks, K GT masks

        Returns:
            max_iou: float, maximum IoU with any GT mask
        """
        if gt_mask.shape[0] == 0 or pred_mask.sum() == 0:
            return 0.0

        # Vectorized computation: compute IoU with all GT masks at once
        pred_sum = pred_mask.sum().float()  # Scalar
        gt_sums = gt_mask.sum(dim=1).float()  # [K]

        # Compute intersection: [K, N] * [N] -> broadcast -> [K, N] -> sum(dim=1) -> [K]
        intersections = (gt_mask * pred_mask).sum(dim=1).float()  # [K]

        # Compute union: [K]
        unions = pred_sum + gt_sums - intersections  # [K]

        # Compute IoU: [K]
        ious = intersections / (unions + 1e-5)  # [K]

        # Return maximum IoU
        return ious.max().item()

    def evolve_discern_network(self):
        # evolve discern_net on collected evo PLY data.
        evo_data_dir = os.path.join(
            self.save_path, 'evo_data', f'epoch_{self.current_epoch:03d}')

        if not os.path.exists(evo_data_dir):
            print(f"No evo data directory found: {evo_data_dir}")
            return

        evo_dataset = EvoDatasetFromFiles(
            evo_data_dir, num_point=4096, voxel_size=self.discern_voxel_size, cfg=self.cfg)
        if len(evo_dataset) == 0:
            print(f"Error: No evo data files found in {evo_data_dir}")
            return

        evo_loader = DataLoader(
            evo_dataset,
            batch_size=self.evo_batch_size,
            shuffle=True,
            num_workers=8,
            drop_last=True
        )

        optimizer = optim.Adam(self.discern_net.parameters(),
                               lr=self.evo_lr, weight_decay=1e-4)
        self.discern_net.train()

        if self.discern_backend == "sparseUnet":
            criterion = nn.CrossEntropyLoss()
            for epoch in range(self.evo_epochs_per_batch):
                total_loss = 0.0
                total_point_correct = 0
                total_point_samples = 0
                for batch_idx, (points, masks) in enumerate(evo_loader):
                    points = points.to('cuda').float()
                    masks = masks.to('cuda').long()
                    voxel_data = self.voxelize_batch(
                        points, masks, self.discern_voxel_size, 'cuda')
                    optimizer.zero_grad()
                    logits = self.discern_net(voxel_data)
                    loss = criterion(logits, voxel_data['labels'])
                    loss.backward()
                    optimizer.step()
                    _, _, _, total_correct, total_points = self.calculate_original_point_accuracy(
                        logits, voxel_data)
                    total_loss += loss.item()
                    total_point_correct += total_correct
                    total_point_samples += total_points
                avg_loss = total_loss / len(evo_loader)
                avg_point_accuracy = total_point_correct / total_point_samples
                print(
                    f"Evo Train Epoch {epoch+1}/{self.evo_epochs_per_batch} - Loss: {avg_loss:.4f}, PointAcc: {avg_point_accuracy:.4f}")
        elif self.discern_backend == "pointTransformer":
            criterion = nn.CrossEntropyLoss()
            for epoch in range(self.evo_epochs_per_batch):
                total_loss = 0.0
                total_point_correct = 0
                total_point_samples = 0
                for batch_idx, (points, masks) in enumerate(evo_loader):
                    points = points.to('cuda').float()
                    masks = masks.to('cuda').long()
                    optimizer.zero_grad()
                    point_data = self.prepare_point_transformer_input(
                        points, masks, device='cuda')
                    data_dict = {
                        'coord': point_data['coord'],
                        'feat': point_data['feat'],
                        'offset': point_data['offset']
                    }
                    logits = self.discern_net(data_dict)
                    loss = criterion(logits, point_data['labels'])
                    loss.backward()
                    optimizer.step()
                    point_pred = logits.argmax(-1)
                    total_point_correct += point_pred.eq(
                        point_data['labels']).sum().item()
                    total_point_samples += point_data['labels'].numel()
                    total_loss += loss.item()
                avg_loss = total_loss / len(evo_loader)
                avg_point_accuracy = total_point_correct / \
                    total_point_samples if total_point_samples > 0 else 0.0
                print(
                    f"Evo Train Epoch {epoch+1}/{self.evo_epochs_per_batch} - Loss: {avg_loss:.4f}, PointAcc: {avg_point_accuracy:.4f}")
        elif self.discern_backend == "pointNet":
            criterion = nn.NLLLoss()
            for epoch in range(self.evo_epochs_per_batch):
                total_loss = 0.0
                total_point_correct = 0
                total_point_samples = 0
                for batch_idx, (points, masks) in enumerate(evo_loader):
                    points = points.to('cuda').float()
                    masks = masks.to('cuda').long()
                    optimizer.zero_grad()
                    pred, _ = self.discern_net(points.transpose(2, 1))
                    loss = criterion(pred.reshape(-1, 2), masks.view(-1))
                    loss.backward()
                    optimizer.step()
                    point_pred = pred.argmax(-1)
                    total_point_correct += point_pred.eq(masks).sum().item()
                    total_point_samples += masks.numel()
                    total_loss += loss.item()
                avg_loss = total_loss / len(evo_loader)
                avg_point_accuracy = total_point_correct / \
                    total_point_samples if total_point_samples > 0 else 0.0
                print(
                    f"Evo Train Epoch {epoch+1}/{self.evo_epochs_per_batch} - Loss: {avg_loss:.4f}, PointAcc: {avg_point_accuracy:.4f}")
        else:
            raise ValueError(
                f"evolve_discern_network: unknown discern_backend {self.discern_backend!r}")

        self.loss_dict['evo discern loss'] += avg_loss

        print(
            f"Evo discern training ({self.discern_backend}) done - Final Loss: {avg_loss:.4f}, "
            f"Final PointAcc: {avg_point_accuracy:.4f}, Processed {len(evo_dataset)} evo samples"
        )

        # Save the trained discern_net weights
        evo_ckpts_dir = os.path.join(self.save_path, 'evo_ckpts')
        os.makedirs(evo_ckpts_dir, exist_ok=True)
        discern_net_save_path = os.path.join(
            evo_ckpts_dir, f'discern_net_epoch_{self.current_epoch:03d}.pth')
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.discern_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'point_accuracy': avg_point_accuracy,
            'num_samples': len(evo_dataset)
        }, discern_net_save_path)
        print(f"Saved evolved discern_net weights to: {discern_net_save_path}")

        self.discern_net.eval()

    def intersection_over_union(self, mask1, mask2):
        inter_area = (mask1*mask2).sum()
        union_area = mask1.sum() + mask2.sum() - inter_area
        return inter_area / (union_area + 1e-5)

    def get_maxmatch_mask(self, target_mask, target_mask_completeness, cur_mask):
        ious = []
        for target_mask_id in range(target_mask.shape[-1]):
            inter_area = (cur_mask * target_mask[:, target_mask_id]).sum()
            union_area = cur_mask.sum() + target_mask[:, target_mask_id].sum()/(
                target_mask_completeness[target_mask_id]+1e-5) - inter_area
            iou = inter_area / (union_area + 1e-5)
            completeness = target_mask_completeness[target_mask_id]
            ious.append(iou*completeness)
        return max(ious)

    def train_model(self, epochs):
        train_data_loader = self.train_dataset.get_loader(shuffle=True)
        start = self.load_checkpoint()
        self.refresh_info()
        # Initialize epoch-level pseudo mask statistics
        self.epoch_new_pseudo_predicted = 0
        self.epoch_new_pseudo_gt = 0
        self.epoch_merged_pseudo_predicted = 0
        self.epoch_merged_pseudo_gt = 0
        # Initialize TP counters for each IoU threshold
        self.epoch_new_pseudo_tp = {thr: 0 for thr in self.sufficiency_iou_thresholds}
        self.epoch_merged_pseudo_tp = {thr: 0 for thr in self.sufficiency_iou_thresholds}
        for epoch in range(start, epochs):
            for batch_idx, batch in enumerate(train_data_loader):
                self.train_batch(batch, batch_idx+1, epoch,
                                 len(train_data_loader))

            # Compute sufficiency for epoch (for each IoU threshold)
            self.logger.info('=' * 80)
            self.logger.info(f'Epoch {epoch} Pseudo Mask Statistics:')
            self.logger.info(f'  New pseudo masks:')
            self.logger.info(f'    GT: {self.epoch_new_pseudo_gt}')
            for thr in self.sufficiency_iou_thresholds:
                new_sufficiency = self.epoch_new_pseudo_tp[thr] / (self.epoch_new_pseudo_gt + 1e-5)
                self.logger.info(
                    f'    IoU>{thr:.2f} - TP: {self.epoch_new_pseudo_tp[thr]}, Sufficiency: {new_sufficiency:.4f}'
                )

            self.logger.info(f'  Merged pseudo masks (after deduplication):')
            self.logger.info(f'    GT: {self.epoch_merged_pseudo_gt}')
            for thr in self.sufficiency_iou_thresholds:
                merged_sufficiency = self.epoch_merged_pseudo_tp[thr] / (self.epoch_merged_pseudo_gt + 1e-5)
                self.logger.info(
                    f'    IoU>{thr:.2f} - TP: {self.epoch_merged_pseudo_tp[thr]}, Sufficiency: {merged_sufficiency:.4f}'
                )
            self.logger.info('=' * 80)

            print('=' * 80)
            print(f'Epoch {epoch} Pseudo Mask Statistics:')
            print(f'  New pseudo masks:')
            print(f'    GT: {self.epoch_new_pseudo_gt}')
            for thr in self.sufficiency_iou_thresholds:
                new_sufficiency = self.epoch_new_pseudo_tp[thr] / (self.epoch_new_pseudo_gt + 1e-5)
                print(
                    f'    IoU>{thr:.2f} - TP: {self.epoch_new_pseudo_tp[thr]}, Sufficiency: {new_sufficiency:.4f}'
                )

            print(f'  Merged pseudo masks (after deduplication):')
            print(f'    GT: {self.epoch_merged_pseudo_gt}')
            for thr in self.sufficiency_iou_thresholds:
                merged_sufficiency = self.epoch_merged_pseudo_tp[thr] / (self.epoch_merged_pseudo_gt + 1e-5)
                print(
                    f'    IoU>{thr:.2f} - TP: {self.epoch_merged_pseudo_tp[thr]}, Sufficiency: {merged_sufficiency:.4f}'
                )
            print('=' * 80)

            # Reset epoch statistics for next epoch
            self.epoch_new_pseudo_predicted = 0
            self.epoch_new_pseudo_gt = 0
            self.epoch_merged_pseudo_predicted = 0
            self.epoch_merged_pseudo_gt = 0
            # Reset TP counters for each IoU threshold
            self.epoch_new_pseudo_tp = {thr: 0 for thr in self.sufficiency_iou_thresholds}
            self.epoch_merged_pseudo_tp = {thr: 0 for thr in self.sufficiency_iou_thresholds}

            if epoch in self.evo_epochs:
                # Check if there are evo data files available for training
                evo_data_dir = os.path.join(
                    self.save_path, 'evo_data', f'epoch_{epoch:03d}')
                if os.path.exists(evo_data_dir):
                    ply_files = [f for f in os.listdir(evo_data_dir) if f.endswith(
                        '.ply') and 'input_with_labels' in f]
                    print(
                        f'[Epoch {epoch }] evo collect {len(ply_files)} ply files found')
                    if len(ply_files) >= self.evo_batch_size:
                        self.evolve_discern_network()
                    else:
                        print(
                            f'[Epoch {epoch }] evo collect {len(ply_files)} samples collected (need {self.evo_batch_size}). Skip this round evolution')
                else:
                    print(
                        f'[ERROR] [Epoch {epoch }] no evo data directory found!! skip!!')
            if epoch % 10 == 0:
                self.save_checkpoint(epoch)
                self.validation(vis=False, log=True)

    def save_checkpoint(self, epoch):
        path = os.path.join(self.save_path, 'checkpoint_{}.tar'.format(epoch))
        if not os.path.exists(path):
            torch.save({'epoch': epoch,
                        'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(),
                        'actor_state_dict': self.actor.state_dict(), 'opt_actor_state_dict': self.optimizer_actor.state_dict(),
                        'critic_state_dict': self.critic.state_dict(), 'opt_critic_state_dict': self.optimizer_critic.state_dict(),
                        'discern_net_state_dict': self.discern_net.state_dict(), 'opt_discern_state_dict': self.optimizer_discern.state_dict()
                        }, path)

    def load_checkpoint(self):
        checkpoints = glob(self.save_path+'/*tar')
        if len(checkpoints) == 0:
            print('No checkpoints found at {}'.format(self.save_path))
            return 0

        checkpoints = [os.path.splitext(os.path.basename(path))[
            0].split('_')[-1] for path in checkpoints]
        checkpoints = np.array(checkpoints, dtype=int)
        checkpoints = np.sort(checkpoints)
        path = os.path.join(
            self.save_path, 'checkpoint_{}.tar'.format(checkpoints[-1]))

        print('Loaded checkpoint from: {}'.format(path))
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.optimizer_actor.load_state_dict(
            checkpoint['opt_actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer_critic.load_state_dict(
            checkpoint['opt_critic_state_dict'])

        if 'discern_net_state_dict' in checkpoint:
            self.discern_net.load_state_dict(checkpoint['discern_net_state_dict'])
            print('Loaded discern_net state')
        if 'opt_discern_state_dict' in checkpoint:
            self.optimizer_discern.load_state_dict(
                checkpoint['opt_discern_state_dict'])
            print('Loaded discern optimizer state')

        epoch = checkpoint['epoch']
        return epoch

    def validation(self, vis=True, log=False):
        self.load_checkpoint()
        self.refresh_info()
        self.preds, self.gt = {}, {}
        self.model.eval()
        val_data_loader = self.val_dataset.get_loader(shuffle=False)
        for batch_idx, batch in enumerate(val_data_loader):
            with torch.no_grad():
                coords, feature, normals, target, scene_name, _, _, inverse_map, unique_map, voxl_pc, full_pc, voxl_sp, pointsp, voxl_mysp, exist_pseudo = batch
                batch_sp = [voxl_sp[i].cuda() for i in range(len(voxl_sp))]
                if not coords.is_contiguous():
                    coords = coords.contiguous()
                in_field = spconv.SparseConvTensor(features=feature.cuda(), indices=coords.int().cuda(),
                                                   spatial_shape=list(
                                                       coords.max(0)[0] + 16)[1:],
                                                   batch_size=coords.max(0)[0][0].item() + 1)
                if self.cfg.use_sp:
                    output = self.model(in_field, point2segment=batch_sp,
                                        raw_coordinates=feature[:, -3:].cuda(), train_on_segments=self.cfg.use_sp)
                    sp_score = output["pred_masks"]  # [(bs), N, 10]
                    voxel_masks = sp_score[0][voxl_sp[0]].sigmoid()
                else:
                    output = self.model(
                        in_field, raw_coordinates=feature[:, -3:].cuda(), train_on_segments=self.cfg.use_sp)
                    voxel_score = output["pred_masks"]
                    voxel_masks = voxel_score[0].sigmoid()

                masks = voxel_masks[inverse_map[0]].detach().cpu()
                hard_masks = (masks > 0.5)

                valid_mask_idx, mask_score = [], []

                for mask_id in range(self.model.num_queries):
                    score = masks[:, mask_id][hard_masks[:, mask_id]].mean()
                    if torch.argmax(output["pred_logits"][0][mask_id]) == 0 and hard_masks[:, mask_id].sum() > self.mask_min_size:
                        valid_mask_idx.append(mask_id)
                        mask_score.append(score.item())
                #
                valid_masks = hard_masks[:, valid_mask_idx]
                if len(valid_mask_idx) > 0:
                    pred_instance_color = np.vstack(
                        get_evenly_distributed_colors(valid_masks.shape[1]))

            if vis and len(valid_mask_idx) > 0:
                with torch.no_grad():
                    full_pc = full_pc[0].numpy()
                    os.makedirs(self.cfg.save_path +
                                '/vis_scene', exist_ok=True)
                    predcolor, gtcolor = np.ones_like(
                        full_pc) * 128, np.ones_like(full_pc) * 128
                    for mask_id in range(valid_masks.shape[1]):
                        predcolor2 = np.ones_like(full_pc) * 128
                        mask = valid_masks[:, mask_id]
                        predcolor2[mask] = pred_instance_color[mask_id]
                        predcolor[mask] = pred_instance_color[mask_id]
                        # write_ply(os.path.join(self.cfg.save_path + '/vis', scene_name[0] + 'preds_'+str(mask_id)+'.ply'), [full_pc, predcolor2.astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])
                    pc_centered = full_pc - full_pc.mean(axis=0, keepdims=True)
                    write_ply(os.path.join(self.cfg.save_path + '/vis_scene', scene_name[0] + 'preds_evo_vae.ply'), [
                              pc_centered, predcolor.astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])
                    if len(target[0]['masks']) > 0:
                        gt_instance_color = np.vstack(
                            get_evenly_distributed_colors(len(target[0]['masks'])))
                    for mask_id in range(len(target[0]['masks'])):
                        gtcolor[target[0]['masks'][:, inverse_map[0]]
                                [mask_id] == 1] = gt_instance_color[mask_id]
                    gt_centered = full_pc - full_pc.mean(axis=0, keepdims=True)
                    write_ply(os.path.join(self.cfg.save_path + '/vis_scene', scene_name[0] + 'gt_evo_vae.ply'), [
                              gt_centered, gtcolor.astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])

            self.preds[scene_name[0]] = {"pred_masks": valid_masks.cpu().numpy(), "pred_scores": (torch.tensor(
                mask_score)).cpu().numpy(), "pred_classes": 5 * torch.ones(valid_masks.shape[-1]).cpu().numpy()}
            gt_file = os.path.join(
                self.cfg.data_dir, 'instance_gt', self.val_dataset.mode, scene_name[0] + '.txt')
            self.gt[scene_name[0]] = gt_file
        evaluate(False, self.preds, self.gt, self.logger, log, self.save_path)
        torch.cuda.empty_cache()
        torch.cuda.synchronize(torch.device("cuda"))

    def save_scannet_for_online_test(self):
        self.load_checkpoint()
        self.refresh_info()
        self.preds, self.gt = {}, {}
        self.model.eval()
        val_data_loader = self.val_dataset.get_loader(shuffle=False)
        for batch_idx, batch in enumerate(val_data_loader):
            with torch.no_grad():
                coords, feature, normals, target, scene_name, semantic, instance, inverse_map, unique_map, voxl_pc, full_pc, voxl_sp, pointsp, voxl_mysp, exist_pseudo = batch
                batch_sp = [voxl_sp[i].cuda() for i in range(len(voxl_sp))]
                if not coords.is_contiguous():
                    coords = coords.contiguous()
                in_field = spconv.SparseConvTensor(features=feature.cuda(), indices=coords.int().cuda(),
                                                   spatial_shape=list(
                                                       coords.max(0)[0] + 16)[1:],
                                                   batch_size=coords.max(0)[0][0].item() + 1)

                if self.cfg.use_sp:
                    output = self.model(in_field, point2segment=batch_sp,
                                        raw_coordinates=feature[:, -3:].cuda(), train_on_segments=self.cfg.use_sp)
                    sp_score = output["pred_masks"]  # [(bs), N, 10]
                    voxel_masks = sp_score[0][voxl_sp[0]].sigmoid()
                else:
                    output = self.model(
                        in_field, raw_coordinates=feature[:, -3:].cuda(), train_on_segments=self.cfg.use_sp)
                    voxel_score = output["pred_masks"]
                    voxel_masks = voxel_score[0].sigmoid()

                masks = voxel_masks[inverse_map[0]].detach().cpu()
                hard_masks = (masks > 0.5)

                valid_mask_idx, mask_score = [], []

                for mask_id in range(self.model.num_queries):
                    score = masks[:, mask_id][hard_masks[:, mask_id]].mean()
                    if torch.argmax(output["pred_logits"][0][mask_id]) == 0 and hard_masks[:, mask_id].sum() > 100:
                        valid_mask_idx.append(mask_id)
                        # rec error as maskscore
                        mask_score.append(score.item())
                #
                valid_masks = hard_masks[:, valid_mask_idx]
                if len(valid_masks.shape) == 1:
                    valid_masks = valid_masks.unsqueeze(1)

                os.makedirs(self.cfg.save_path +
                            '/scannet_format', exist_ok=True)
                os.makedirs(self.cfg.save_path +
                            '/scannet_format/predicted_masks', exist_ok=True)
                # save into scannet format
                f = open(os.path.join(self.cfg.save_path +
                         '/scannet_format', f"{scene_name[0]}.txt"), "w")
                cnt = 0
                label_id = 5
                for mask_id in range(valid_masks.shape[1]):
                    single_score = float(mask_score[mask_id])  # float
                    single_mask = valid_masks[:,
                                              mask_id].numpy().astype(np.uint8)

                    rel_fn = f"predicted_masks/{scene_name[0]}_{cnt:03d}.txt"
                    # meta file
                    f.write(f"{rel_fn} {label_id} {single_score:.4f}\n")
                    mask_path = os.path.join(
                        self.cfg.save_path + '/scannet_format', rel_fn)
                    np.savetxt(mask_path, single_mask, fmt="%d")
                    cnt += 1
                f.close()

        torch.cuda.empty_cache()
        torch.cuda.synchronize(torch.device("cuda"))

    def extract_shape_pts(self, decoder, embedding, N=32, sample_pts_num=10000):
        bs = embedding[0].shape[0]
        validness = []
        max_batch = self.sdf_bs
        voxel_origin = [-1.0, -1.0, -1.0]
        voxel_size = 2.0 / (N)  # why minus 1?
        overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())

        samples = torch.zeros(N ** 3, 4)
        samples[:, 2] = overall_index % N
        samples[:, 1] = (overall_index.long() / N) % N
        samples[:, 0] = ((overall_index.long() / N) / N) % N

        samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
        samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
        samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]
        samples = samples.unsqueeze(0).repeat(bs, 1, 1)

        head, num_samples = 0, N ** 3

        with torch.no_grad():
            while head < bs:
                tmp_embedding_0 = embedding[0][head: min(
                    head + max_batch, num_samples)]
                tmp_embedding_1 = embedding[1][head: min(
                    head + max_batch, num_samples)]
                sample_subset = samples[head: min(
                    head + max_batch, num_samples), :, 0:3].cuda()
                samples[head: min(head + max_batch, num_samples), :, 3] = decoder(
                    sample_subset/2, tmp_embedding_0, tmp_embedding_1).squeeze(1).detach().cpu().float()
                head += max_batch
        sdf_values = samples[:, :, 3]

        onsurf_points_list = []
        for b in range(bs):
            sdf_value = sdf_values[b].reshape(N, N, N)
            try:
                verts, faces, normals, values = skimage.measure.marching_cubes(
                    sdf_value.numpy(), level=0.0, spacing=[voxel_size] * 3)
                mesh_points = np.zeros_like(verts)
                mesh_points[:, 0] = voxel_origin[0] + verts[:, 0]
                mesh_points[:, 1] = voxel_origin[1] + verts[:, 1]
                mesh_points[:, 2] = voxel_origin[2] + verts[:, 2]
                mesh_points /= 2

                onsurf_points = torch.from_numpy(sample_points_from_mesh(
                    mesh_points, faces, sample_pts_num)).float().cuda()
                if onsurf_points.shape[0] != sample_pts_num:
                    onsurf_points = onsurf_points[np.random.choice(
                        onsurf_points.shape[0], sample_pts_num, replace=True)]
                onsurf_points_list.append(onsurf_points.unsqueeze(0))
                validness.append(True)
            except:
                validness.append(False)
                print('cannot recovery')
        try:
            if len(onsurf_points_list) > 0:
                return torch.cat(onsurf_points_list), validness
            else:
                return [], validness
        except:
            print(1)

    def generate_mesh(self, decoder, embedding, N=32, path=None):
        voxel_origin = [-1.0, -1.0, -1.0]
        voxel_size = 2.0 / (N)  # why minus 1?
        overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())

        samples = torch.zeros(N ** 3, 4)
        samples[:, 2] = overall_index % N
        samples[:, 1] = (overall_index.long() / N) % N
        samples[:, 0] = ((overall_index.long() / N) / N) % N

        samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
        samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
        samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]
        samples = samples.unsqueeze(0)

        with torch.no_grad():
            sample_subset = samples[:, :, 0:3].cuda()
            samples[:, :, 3] = decoder(
                sample_subset / 2, *embedding).squeeze(1).detach().cpu().float()
        sdf_values = samples[:, :, 3]

        sdf_value = sdf_values.reshape(N, N, N)
        verts, faces, normals, values = skimage.measure.marching_cubes(
            sdf_value.numpy(), level=0.0, spacing=[voxel_size] * 3)
        mesh_points = np.zeros_like(verts)
        mesh_points[:, 0] = voxel_origin[0] + verts[:, 0]
        mesh_points[:, 1] = voxel_origin[1] + verts[:, 1]
        mesh_points[:, 2] = voxel_origin[2] + verts[:, 2]
        mesh_points /= 2
        num_verts = verts.shape[0]
        num_faces = faces.shape[0]
        verts_tuple = np.zeros((num_verts,), dtype=[
                               ("x", "f4"), ("y", "f4"), ("z", "f4")])
        for i in range(0, num_verts):
            verts_tuple[i] = tuple(mesh_points[i, :])
        faces_building = []
        for i in range(0, num_faces):
            faces_building.append(((faces[i, :].tolist(),)))
        faces_tuple = np.array(faces_building, dtype=[
                               ("vertex_indices", "i4", (3,))])
        el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
        el_faces = plyfile.PlyElement.describe(faces_tuple, "face")
        ply_data = plyfile.PlyData([el_verts, el_faces])
        logging.debug("saving mesh to %s" % ('tmp.ply'))
        ply_data.write(path)
        return torch.from_numpy(sample_points_from_mesh(mesh_points, faces, self.convergence_sample_num)).float().cuda()


def sample_points_from_mesh(vertices, faces, num_samples):
    def compute_area(vertices, faces):
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        cross_product = np.cross(v1 - v0, v2 - v0)
        area = np.linalg.norm(cross_product, axis=1) * 0.5
        return area

    def sample_faces(faces, areas, num_samples):
        cumulative_areas = np.cumsum(areas)
        cumulative_areas /= cumulative_areas[-1]
        samples = np.random.rand(num_samples)
        face_indices = np.searchsorted(cumulative_areas, samples)
        return face_indices

    def sample_points(vertices, faces, face_indices):
        v0 = vertices[faces[face_indices, 0]]
        v1 = vertices[faces[face_indices, 1]]
        v2 = vertices[faces[face_indices, 2]]

        u = np.random.rand(len(face_indices), 1)
        v = np.random.rand(len(face_indices), 1)
        is_above = (u + v) > 1
        u[is_above] = 1 - u[is_above]
        v[is_above] = 1 - v[is_above]

        sampled_points = (1 - u - v) * v0 + u * v1 + v * v2
        return sampled_points

    areas = compute_area(vertices, faces)
    face_indices = sample_faces(faces, areas, num_samples)
    sampled_points = sample_points(vertices, faces, face_indices)

    return sampled_points


def remove_duplications(masks, score, iou_th=0.5, inclusion_flag=True, inclusion_th=0.8):
    # masks: [N, K]
    # scores: [K]
    N = masks.shape[-1]
    active_mask = torch.ones(N).to(masks.device)
    for i in range(N):
        if active_mask[i] == 0:
            continue  # if removed already
        # find duplication
        '''here B can be replaced by ppt.trj[w]???'''
        B = masks * active_mask[None, :]
        '''D is the raw sdf, get its valid proposal, and compute iou for each proposal with all others'''
        inter = torch.logical_and(B, B[:, i: i + 1])
        union = torch.logical_or(B, B[:, i: i + 1])
        iou = inter.sum(0).float() / (union.sum(0).float() + 1e-6)
        duplication_mask = iou >= iou_th
        '''for each proposal, identify all duplications, only retain the highest score one, may delete cur proposal itself'''
        # merge
        if duplication_mask.sum() > 1:
            _score = score.clone()
            _score[~duplication_mask] = 0.0
            merge_to_i = _score.argmax()
            active_mask[duplication_mask] = 0.0
            active_mask[merge_to_i] = 1.0

    if inclusion_flag:
        for i in range(N):
            if active_mask[i] == 0:
                continue  # if removed already
            # find duplication
            B = masks * active_mask[None, :]
            inter = torch.logical_and(B, B[:, i: i + 1])
            ratio = inter.sum(0).float() / \
                (B[:, i: i + 1].sum().float() + 1e-6)
            ratio[i] = 0.0
            inclusion_ratio = ratio.max()
            if inclusion_ratio > inclusion_th:  # reject
                active_mask[i] = 0.0

    return torch.where(active_mask == 1)[0]


def compute_dice_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def compute_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none")
    # loss = F.binary_cross_entropy(inputs, targets, reduction="none")
    return loss.mean(1).sum() / num_masks
