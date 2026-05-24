from __future__ import print_function

import glob
import math
import os
import random

import numpy as np
import torch
import torch.utils.data as data

from .build import DATASETS


@DATASETS.register_module()
class ShapeNet6clsDataset(data.Dataset):
    """ShapeNet 6-class dataset with partial and complete point clouds."""

    def __init__(self, args):
        self.args = args

        default_root = '/media/SSD/zihui/simon/data/shapenet_rendered'
        self.data_root = getattr(args, 'DATA_ROOT', None) or os.environ.get('SHAPENET_MULTI_ROOT', default_root)
        self.split_root = getattr(args, 'SPLIT_ROOT', None) or os.environ.get('SHAPENET_MULTI_SPLIT_ROOT', os.path.join(self.data_root, 'shapenet_splits'))

        self.class_choice = self.args.category
        self.split = self.args.subset

        self.class_mapping = {
            '02691156': 'plane',
            '02933112': 'cabinet',
            '03001627': 'chair',
            '04090263': 'rifle',
            '04256520': 'sofa',
            '04401088': 'telephone',
        }
        self.name_to_id = {v: k for k, v in self.class_mapping.items()}

        self.samples = []
        if self.class_choice == 'all':
            for class_name, class_id in self.name_to_id.items():
                self.samples.extend(self._load_class_samples(class_id, class_name))
        else:
            key = self.class_choice.lower()
            if key not in self.name_to_id:
                raise ValueError(f'Unknown class: {self.class_choice}')
            class_id = self.name_to_id[key]
            self.samples.extend(self._load_class_samples(class_id, key))

    def _load_class_samples(self, class_id, class_name):
        samples = []
        split_file = os.path.join(self.split_root, class_id, f'{self.split}.lst')
        if not os.path.exists(split_file):
            print(f'Warning: Split file {split_file} not found')
            return samples

        with open(split_file, 'r') as f:
            sample_ids = [line.strip() for line in f.readlines()]

        class_data_dir = os.path.join(self.data_root, f'{class_id}_dep')
        for sample_id in sample_ids:
            sample_dir = os.path.join(class_data_dir, sample_id)
            if not os.path.exists(sample_dir):
                continue
            full_mesh_path = os.path.join(sample_dir, 'full_mesh_pcl.npz')
            if not os.path.exists(full_mesh_path):
                continue

            partial_files = glob.glob(os.path.join(sample_dir, 'dep_pcl_*.npz'))
            if len(partial_files) == 0:
                continue

            samples.append(
                {
                    'sample_id': sample_id,
                    'class_id': class_id,
                    'class_name': class_name,
                    'sample_dir': sample_dir,
                    'full_mesh_path': full_mesh_path,
                    'partial_files': partial_files,
                    'label': list(self.class_mapping.keys()).index(class_id),
                }
            )
        return samples

    @staticmethod
    def get_rotation_matrix(axis, angle, device):
        if axis == 'z':
            return torch.tensor(
                [[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]],
                dtype=torch.float32,
                device=device,
            )
        if axis == 'y':
            return torch.tensor(
                [[math.cos(angle), 0, math.sin(angle)], [0, 1, 0], [-math.sin(angle), 0, math.cos(angle)]],
                dtype=torch.float32,
                device=device,
            )
        if axis == 'x':
            return torch.tensor(
                [[1, 0, 0], [0, math.cos(angle), -math.sin(angle)], [0, math.sin(angle), math.cos(angle)]],
                dtype=torch.float32,
                device=device,
            )
        raise ValueError(f'Unknown axis {axis}')

    @staticmethod
    def _resample_points(points_np, target_num_points=1024):
        if points_np.shape[0] > target_num_points:
            idx = np.random.choice(points_np.shape[0], target_num_points, replace=False)
            return points_np[idx]
        if points_np.shape[0] < target_num_points:
            idx = np.random.choice(points_np.shape[0], target_num_points, replace=True)
            return points_np[idx]
        return points_np

    def __getitem__(self, index):
        sample = self.samples[index]

        available_files = sample['partial_files']
        num_views = min(random.randint(2, 4), len(available_files))
        selected_files = random.sample(available_files, num_views)

        partial_clouds = []
        for partial_file in selected_files:
            partial_data = np.load(partial_file)
            partial_clouds.append(partial_data['p_w'].astype(np.float32))
        partial_np = np.concatenate(partial_clouds, axis=0)

        full_data = np.load(sample['full_mesh_path'])
        gt_np = full_data['p_w'].astype(np.float32)

        partial_np = self._resample_points(partial_np, 1024)
        gt_np = self._resample_points(gt_np, 1024)

        label = sample['label']
        gt = torch.from_numpy(gt_np)
        partial = torch.from_numpy(partial_np)

        device = gt.device
        angle_z = math.radians(random.uniform(-180, 180))
        angle_y = math.radians(random.uniform(-10, 10))
        angle_x = math.radians(random.uniform(-10, 10))
        r = (
            self.get_rotation_matrix('z', angle_z, device)
            @ self.get_rotation_matrix('y', angle_y, device)
            @ self.get_rotation_matrix('x', angle_x, device)
        )

        gt = torch.bmm(gt.unsqueeze(0), r.transpose(0, 1).unsqueeze(0)).squeeze(0)
        partial = torch.bmm(partial.unsqueeze(0), r.transpose(0, 1).unsqueeze(0)).squeeze(0)

        center = partial.mean(dim=0)
        radius = torch.norm(partial - center, dim=1).max() + 1e-6
        partial_norm = (partial - center) / radius
        gt_norm = (gt - center) / radius

        return int(label), int(index), partial_norm, gt_norm

    def __len__(self):
        return len(self.samples)
