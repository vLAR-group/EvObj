
import json
import os
import pickle
import sys
from glob import glob

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset

from lib.aug_tools import rota_coords, scale_coords
from lib.helper_ply import read_ply, write_ply


class VoxelizedDataset(Dataset):
    def __init__(self, mode, cfg, data_path, batch_size, num_workers, voxel_size, RL=False):
        self.path = data_path
        self.mode = mode
        self.cfg = cfg
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.voxel_size = voxel_size

        self.ignore_label = getattr(cfg, "ignore_label", -1)
        self.limit_numpoints = getattr(cfg, "limit_numpoints", 9999999999)

        self.scene_prefix = getattr(cfg, "scene_prefix", "")
        if not hasattr(cfg, "class_names_path"):
            raise AttributeError("cfg must have class_names_path attribute")
        self.class_names_path = cfg.class_names_path

        self.data = self._build_scene_list(mode)
        if RL:
            self.data = self.data[0:10]

        # Color normalization:
        # ScanNet-style uses dataset-level mean/std in [0,1] (computed on RGB/255) and then standardizes.
        # For ScanNet++, if you don't have precomputed mean/std, we compute per-scene mean/std on-the-fly
        # (still in [0,1] space) inside __getitem__.
        self.rota_coords = rota_coords(rotation_bound=((-0, 0), (-0, 0), (-np.pi, np.pi)))
        self.scale_coords = scale_coords(scale_bound=(0.9, 1.1))

        self.class_names = self._load_class_names()
        self.class_mapping = {name: idx for idx, name in enumerate(self.class_names)}
        self.chair_whitelist = {
            "office chair",
            "chair",
            "arm chair",
            "sofa chair",
            "dining chair",
            "armchair",
            "lounge chair",
            "office visitor chair",
            "rolling chair",
            "chairs",
            "wheelchair",
            "barber chair",
            "papasan chair",
            "ottoman chair",
            "folding chair",
            "stack of chairs",
            "deck chair",
            "high chair",
            "piano chair",
            "baby chair",
            "bean bag chair",
            "desk chair",
            "gamer chair",
            "gaming chair",
            "hanging chair",
            "relaxing chair",
            "roman chair",
            "round chair",
            "wheeled chair",
        }
        self.chair_label_ids_post = {
            self.class_mapping[name] for name in self.chair_whitelist if name in self.class_mapping
        }

    def __len__(self):
        return len(self.data)

    def _load_class_names(self):
        if os.path.exists(self.class_names_path):
            with open(self.class_names_path) as f:
                class_names = [line.strip() for line in f if line.strip()]
        else:
            class_names = []
        return class_names



    def _build_scene_list(self, mode):
        # No fallback: require splits_dir and fixed filenames
        splits_dir = getattr(self.cfg, "splits_dir", None)
        if not splits_dir:
            raise ValueError("cfg.splits_dir is required (e.g. /.../scannet_pp/splits)")

        split_name = {
            "train": "nvs_sem_train.txt",
            "validation": "nvs_sem_val.txt",
            "test": "nvs_sem_test.txt",
        }[mode]
        split_file = os.path.join(splits_dir, split_name)
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file) as f:
            scene_list = [line.strip() for line in f if line.strip()]

        data = []
        for scene_id in scene_list:
            scene_root = os.path.join(self.path, scene_id, "scans")
            mesh_path = os.path.join(scene_root, "mesh_aligned_0.05.ply")
            segments_path = os.path.join(scene_root, "segments.json")
            anno_path = os.path.join(scene_root, "segments_anno.json")
            if os.path.exists(mesh_path) and os.path.exists(segments_path) and os.path.exists(anno_path):
                data.append(
                    {
                        "scene_id": scene_id,
                        "mesh_path": mesh_path,
                        "segments_path": segments_path,
                        "anno_path": anno_path,
                    }
                )
        return data

    def _load_json(self, filepath):
        with open(filepath, "r") as f:
            return json.load(f)

    def _map_semantic_and_instance(self, segment_ids, seg_groups):
        num_vertices = len(segment_ids)
        semantic = np.full((num_vertices, 1), self.ignore_label, dtype=np.int32)
        instance = np.full((num_vertices, 1), self.ignore_label, dtype=np.int32)

        next_instance_id = 0
        for group in seg_groups:
            label_name = group.get("label")
            if label_name not in self.chair_whitelist:
                continue
            label = self.class_mapping.get(label_name, self.ignore_label)
            if label == self.ignore_label:
                continue

            segments = group.get("segments", [])
            if len(segments) == 0:
                continue

            mask = np.isin(segment_ids, segments)
            semantic[mask] = label
            instance[mask] = next_instance_id
            next_instance_id += 1

        return semantic, instance

    def __getitem__(self, idx):
        sample = self.data[idx]

        ply = read_ply(sample["mesh_path"], triangular_mesh=True)
        ply = ply[0]
        pc = np.vstack((ply["x"], ply["y"], ply["z"])).T.astype(np.float32)
        color = np.vstack((ply["red"], ply["green"], ply["blue"])).T.astype(np.float32)
        normals = np.zeros_like(pc)
        segments = self._load_json(sample["segments_path"])
        
        anno = self._load_json(sample["anno_path"])
        segment_ids = np.array(segments["segIndices"], dtype=np.int32)
        semantic, instance = self._map_semantic_and_instance(segment_ids, anno["segGroups"])

        pc = pc - pc.min(0)

        if self.mode == "train":
            pc[:, 0:2] = pc[:, 0:2] + (np.random.uniform(pc.min(0), pc.max(0)) / 2)[0:2][None, ...]
            for i in (0, 1):
                if np.random.random() < 0.5:
                    pc_max = np.max(pc[:, i])
                    pc[:, i] = pc_max - pc[:, i]
            pc = self.scale_coords(pc)
            pc = self.rota_coords(pc)

        # Keep numeric types consistent (avoid float64 promotion from numpy ops).
        pc = pc.astype(np.float32, copy=False)

        # Per-scene color standardization (RGB is in [0, 255]).
        color_mean = color.mean(axis=0, dtype=np.float32)
        color_std = color.std(axis=0, dtype=np.float32)
        color_std = np.maximum(color_std, 1e-6).astype(np.float32, copy=False)
        color = ((color - color_mean) / color_std).astype(np.float32, copy=False)

        feature = np.concatenate([color, pc], 1).astype(np.float32, copy=False)
        coords, feature, semantic, instance, unique_map, inverse_map = self.voxelize(pc, feature, semantic, instance)

        feature = torch.from_numpy(feature)
        semantic = torch.from_numpy(semantic)
        instance = torch.from_numpy(instance)
        pc_full = torch.from_numpy(pc)
        pc = torch.from_numpy(pc[unique_map])
        normals = F.normalize(torch.from_numpy(normals), dim=-1)

        scene_name = f"{self.scene_prefix}{sample['scene_id']}"

        mysp_voxel = segment_ids[unique_map]

        sp_idx = segment_ids
        sp_idx_voxel = sp_idx[unique_map]
        sp_idx_voxel_copy = -np.ones_like(sp_idx_voxel)
        valid_sp_idx = sp_idx_voxel[sp_idx_voxel != -1]
        unique_vals = np.unique(valid_sp_idx)
        unique_vals.sort()
        sp_idx_voxel_copy[sp_idx_voxel != -1] = np.searchsorted(unique_vals, valid_sp_idx)
        sp_idx_voxel = sp_idx_voxel_copy

        exist_mask_file = os.path.join(self.cfg.save_path, "exist_pseudo", scene_name + ".pickle")
        if os.path.exists(exist_mask_file):
            try:
                with open(exist_mask_file, "rb") as f:
                    data = pickle.load(f)
                exist_mask = [data[0][unique_map], data[1]]
            except Exception:
                print("removing: ", exist_mask_file)
                os.system(f"rm -r {exist_mask_file}")
                exist_mask = [torch.zeros((len(semantic), 1)).bool(), torch.tensor(0).unsqueeze(-1)]
        else:
            exist_mask = [torch.zeros((len(semantic), 1)).bool(), torch.tensor(0).unsqueeze(-1)]

        return (
            coords,
            feature,
            normals,
            semantic.squeeze(),
            instance.squeeze(),
            inverse_map,
            unique_map,
            scene_name,
            pc,
            pc_full,
            torch.from_numpy(sp_idx_voxel).long(),
            torch.from_numpy(sp_idx).long(),
            torch.from_numpy(mysp_voxel).long(),
            exist_mask,
        )

    def get_loader(self, shuffle=True):
        return torch.utils.data.DataLoader(
            self,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            shuffle=shuffle,
            worker_init_fn=self.worker_init_fn,
        )

    def worker_init_fn(self, worker_id):
        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)

    def voxelize(self, coords, feature, semantic, instance):
        scale = 1 / self.voxel_size
        coords = coords - coords.min(0)
        coords = np.floor(coords * scale)
        coords, unique_map, inverse_map = np.unique(coords, return_index=True, return_inverse=True, axis=0)
        return coords, feature[unique_map], semantic[unique_map], instance[unique_map], unique_map, inverse_map

    def collate_fn(self, batch):
        coords, feature, normals, semantic, instance, inverse_map, unique_map, scene_name, pc, pc_full, sp_idx, sp_idx_full, mysp, exist_mask = list(
            zip(*batch)
        )
        coords_batch, feature_batch, instance_batch, pc_batch, pc_batch_full, sp_batch, sp_batch_full = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        mysp_batch = []
        target = []
        semantic_batch = []
        normals_batch = []
        exist_mask_batch = []
        batch_num_points = 0
        for batch_id, _ in enumerate(coords):
            num_points = coords[batch_id].shape[0]
            batch_num_points += num_points
            if self.limit_numpoints and batch_num_points > self.limit_numpoints:
                num_full_points = sum(len(c) for c in coords)
                num_full_batch_size = len(coords)
                print(
                    f"\t\tCannot fit {num_full_points} points into {self.limit_numpoints} points "
                    f"limit. Truncating batch size at {batch_id} out of {num_full_batch_size} with {batch_num_points - num_points}."
                )
                break
            coords_batch.append(
                torch.cat((torch.ones(num_points, 1).int() * batch_id, torch.from_numpy(coords[batch_id]).int()), 1)
            )
            feature_batch.append((feature[batch_id]))
            normals_batch.append(normals[batch_id])
            pc_batch.append(pc[batch_id])
            pc_batch_full.append(pc_full[batch_id])
            sp_batch.append(sp_idx[batch_id])
            sp_batch_full.append(sp_idx_full[batch_id])
            mysp_batch.append(mysp[batch_id])
            exist_mask_batch.append(exist_mask[batch_id])

            instance_batch.append(instance[batch_id])
            semantic_batch.append((semantic[batch_id]))
            target.append(dict())
            masks, labels, segment_masks = [], [], []

            valid_sp_mask = sp_idx[batch_id] != -1
            _, ret_index, ret_inv = np.unique(sp_idx[batch_id][valid_sp_mask].numpy(), return_index=True, return_inverse=True)

            sp_instance_label = instance[batch_id][ret_index]
            for instance_id in torch.unique(instance[batch_id]):
                if instance_id == -1:
                    continue

                mask = (instance[batch_id] == instance_id).bool()
                tmp_semantic = semantic[batch_id]
                label = torch.mode(tmp_semantic[mask]).values

                label_value = int(label.item())
                if label_value in self.chair_label_ids_post:
                    masks.append(mask.unsqueeze(0))
                    labels.append(torch.zeros_like(label.unsqueeze(0).long()))
                    segment_masks.append((sp_instance_label == instance_id).bool().unsqueeze(0))

            if len(masks) > 0:
                target[batch_id]["labels"] = torch.cat(labels)
                target[batch_id]["masks"] = torch.cat(masks, dim=0).squeeze(-1)
                target[batch_id]["segment_mask"] = torch.cat(segment_masks, dim=0).squeeze(-1)
            else:
                target[batch_id]["labels"] = []
                target[batch_id]["masks"] = torch.zeros_like(instance[batch_id])[None, :]
                target[batch_id]["segment_mask"] = []

        coords_batch = torch.cat(coords_batch, 0)
        feature_batch = torch.cat(feature_batch, 0).float()
        return (
            coords_batch,
            feature_batch,
            normals_batch,
            target,
            scene_name,
            semantic_batch,
            instance_batch,
            inverse_map,
            unique_map,
            pc_batch,
            pc_batch_full,
            sp_batch,
            sp_batch_full,
            mysp_batch,
            exist_mask_batch,
        )



