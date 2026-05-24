import pathlib
import numpy as np
from torch.utils.data import Dataset
from plyfile import PlyData, PlyElement


class ShapeNetDataset(Dataset):
    def __init__(self,
                 data_root,
                 mode='train',
                 split_txt_dir='txt',
                 num_point=4096,
                 outlier_ratio=0.1,
                 outlier_prob=1,
                 floor_prob=0.1,
                 vertical_plane_prob=0.2,
                 spherical_crop_prob=0.5,
                 other_object_prob=0.7,scale_min=0.8,scale_max=1.5,
                 horizontal_plane_prob=0.1):
        super().__init__()

        self.data_root = pathlib.Path(data_root).expanduser()
        if not self.data_root.is_dir():
            raise FileNotFoundError(f"data_root not found: {self.data_root}")

        # Load train/test split list
        split_path = pathlib.Path(split_txt_dir).expanduser() / f"{mode}.txt"
        if not split_path.is_file():
            raise FileNotFoundError(f"Split file not found: {split_path}")
        with open(split_path, 'r') as f:
            self.split_names = set(line.strip() for line in f if line.strip())

        self.num_point = num_point
        self.outlier_ratio = outlier_ratio
        self.outlier_prob = outlier_prob
        self.floor_prob = floor_prob
        self.vertical_plane_prob = vertical_plane_prob
        self.spherical_crop_prob = spherical_crop_prob
        self.scale_min = scale_min
        self.scale_max = scale_max
        print(f"scale_min: {scale_min}, scale_max: {scale_max}")
        self.other_object_prob = other_object_prob
        self.horizontal_plane_prob = horizontal_plane_prob

        # Read and filter subfolders, only keep scenes in split list
        all_scenes = [d for d in self.data_root.iterdir() if d.is_dir()]
        self.scene_folders = [d for d in all_scenes if d.name in self.split_names]
        if not self.scene_folders:
            raise RuntimeError(f"No scenes found for mode={mode} in {self.data_root}")

        # Collect npz files
        self.npz_files = {}
        for scene in self.scene_folders:
            files = sorted(scene.glob("dep_pcl_*.npz"))
            if len(files) < 3:
                raise RuntimeError(f"Scene '{scene.name}' has fewer than 3 point cloud files.")
            self.npz_files[scene.name] = files

        print(f"[INFO] Mode: {mode} | Sample count: {len(self.npz_files)} | num_point: {self.num_point}")

    def __len__(self):
        return len(self.npz_files)

    def generate_horizontal_plane(self, num_points, height, max_radius):
        # Randomly determine plane size
        plane_size = np.random.uniform(0.8 * max_radius, 1.3 * max_radius)
        half = plane_size / 2
        x = np.random.uniform(-half, half, num_points)
        y = np.random.uniform(-half, half, num_points)
        z = np.full(num_points, height)

        # Combine into (N,3) array
        pts = np.column_stack((x, y, z))

        theta = np.random.uniform(0, 2 * np.pi)
        c, s = np.cos(theta), np.sin(theta)
        Rz = np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
        pts = pts @ Rz.T

        return pts
    def generate_floor(self, num_points, floor_z, max_radius):
        # Randomly generate floor and add random defects
        scale_factor = np.random.uniform(1.0, 1.5)
        floor_size = max_radius * scale_factor
        x = np.random.uniform(-floor_size / 2, floor_size / 2, num_points)
        y = np.random.uniform(-floor_size / 2, floor_size / 2, num_points)
        z = np.full(num_points, floor_z)
        floor_points = np.column_stack((x, y, z))

        # Randomly create holes
        num_holes = np.random.randint(3, 7)
        for _ in range(num_holes):
            # Calculate hole size
            hole_radius = np.random.uniform(floor_size * 0.05, floor_size * 0.1)
            cx, cy = np.random.uniform(-floor_size / 2, floor_size / 2, 2)
            dists = np.sqrt((floor_points[:, 0] - cx) ** 2 + (floor_points[:, 1] - cy) ** 2)
            floor_points = floor_points[dists > hole_radius]
            if floor_points.shape[0] < 10:
                break

        return floor_points

    def generate_vertical_plane(self, num_points, floor_z, max_radius, object_height, all_points):        
        
        # Generate vertical plane
        xy_points = all_points[:, :2]
        distances = np.linalg.norm(xy_points, axis=1)
        max_distance_idx = np.argmax(distances)
        # Find the farthest point in xy plane
        max_x, max_y = xy_points[max_distance_idx]
        width = np.random.uniform(0.3 * max_radius, 2 * max_radius)
        length = np.random.uniform(0.8 * object_height, 2 * object_height)
        offset = np.random.uniform(0.1, 0.3) * max_radius
        # Offset outward from the farthest point
        x = np.full(num_points, max_x + offset)
        # Wall width
        y = np.random.uniform(max_y - width / 2, max_y + width / 2, num_points)
        z_min = floor_z
        z_max = floor_z + length
        z = np.random.uniform(z_min, z_max, num_points)
        plane_points = np.column_stack((x, y, z))
        return plane_points

    def rotate_points(self, points):
        # Randomly rotate point cloud data
        # z-axis 360 degrees, x-axis 5 degrees, y-axis 5 degrees
        angle_z = np.random.uniform(-180, 180) * np.pi / 180
        angle_x = np.random.uniform(-5, 5) * np.pi / 180
        angle_y = np.random.uniform(-5, 5) * np.pi / 180
        R_z = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                        [np.sin(angle_z), np.cos(angle_z), 0],
                        [0, 0, 1]])
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(angle_x), -np.sin(angle_x)],
                        [0, np.sin(angle_x), np.cos(angle_x)]])
        R_y = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                        [0, 1, 0],
                        [-np.sin(angle_y), 0, np.cos(angle_y)]])
        R = np.dot(R_z, np.dot(R_x, R_y))
        rotated_points = np.dot(points, R.T)
        return rotated_points

    def apply_random_spherical_crop(self, points, mask, num_crop_min=2, num_crop_max=4):
        # Random spherical cropping
        # Randomly sample 2-4 points as centers within point cloud range, 
        # crop spherical regions with radius 0.05-0.2, remove points within spherical regions
        num_crop = np.random.randint(num_crop_min, num_crop_max + 1)
        for _ in range(num_crop):
            center = np.random.uniform(np.min(points, axis=0), np.max(points, axis=0), 3)
            radius = np.random.uniform(0.05, 0.2)
            distances = np.linalg.norm(points - center, axis=1)
            mask = mask[distances > radius]
            points = points[distances > radius]
        return points, mask

    def apply_bounding_cylinder_crop(self, points, mask, xy_radius):
        """
        Apply bounding cylinder cropping to point cloud, keep points within cylinder.
        First calculate the farthest distance from points to Z-axis as base radius, then generate a random radius based on that distance.
        The new cylinder center point is randomly offset around xy plane, finally crop based on this new cylinder.
        """
        # Calculate distance from each point to Z-axis (distance from points to origin in xy plane)
        distances_to_z_axis = np.linalg.norm(points[:, :2], axis=1)

        # Use maximum distance as cylinder radius
        max_distance = np.max(distances_to_z_axis)
        # Set a random radius based on maximum distance, range is 0.9 to 1.1 times
        random_radius_factor = np.random.uniform(0.9, 1.1)
        xy_radius = max_distance * random_radius_factor  # Update radius

        # Randomly offset center point's xy values (let center point vary around xy plane)
        xy_offset = np.random.uniform(-0.05, 0.05, 2)  # Random offset range, 2D offset
        center_xy = xy_offset  # New center point offset in xy plane

        # New center point position
        center = np.array([center_xy[0], center_xy[1], 0])  # Keep Z position at 0

        # Calculate distance from each point to new cylinder center (only consider xy plane)
        distances_to_center = np.linalg.norm(points[:, :2] - center[:2], axis=1)

        # Keep points within cylinder
        mask = mask[distances_to_center <= xy_radius]
        points = points[distances_to_center <= xy_radius]

        return points, mask
    def apply_random_scaling(self, points):
        """
        Randomly scale point cloud data.
        """
        # scale_factor = np.random.uniform(0.8, 1.5)
        scale_factor = np.random.uniform(self.scale_min, self.scale_max)
        scaled_points = points * scale_factor
        return scaled_points


    def sample_to_fixed_num(self, points, mask, num_points=4096):
        # If point cloud count exceeds num_points, randomly sample num_points points
        if points.shape[0] > num_points:
            sampled_indices = np.random.choice(points.shape[0], num_points, replace=False)
            points = points[sampled_indices]
            mask = mask[sampled_indices]
        # If point cloud count is less than num_points, pad
        elif points.shape[0] < num_points:
            diff = num_points - points.shape[0]
            padding_points = np.zeros((diff, 3))  # Pad with zero points
            padding_mask = np.zeros(diff, dtype=np.int32)  # Pad mask
            points = np.vstack([points, padding_points])
            mask = np.hstack([mask, padding_mask])
        return points, mask

    def __getitem__(self, idx):
        scene_name = list(self.npz_files.keys())[idx]
        npz_files = self.npz_files[scene_name]

        # Randomly select 3-6 viewpoints
        num_views = np.random.randint(3, 7)
        selected_files = np.random.choice(npz_files, num_views, replace=False)

        points_list = []
        mask_list = []

        # Sample point clouds from each viewpoint and mark foreground
        for f in selected_files:
            data = np.load(f)
            pts = data['p_w']
            N = pts.shape[0]
            if N == 0:
                print(f'ERROR no points in {scene_name}, skipping this file')
                continue  # Skip this empty file
            idxs = np.random.choice(N, self.num_point, replace=(N < self.num_point))
            sampled = pts[idxs]
            points_list.append(sampled)
            mask_list.append(np.ones(self.num_point, dtype=np.int32))

        # Check if we have any valid point clouds
        if len(points_list) == 0:
            print(f'ERROR: No valid point clouds found for scene {scene_name}, trying next sample')
            return self.__getitem__(idx + 1)  # Try next sample
        
        # Concatenate point clouds from all viewpoints
        points = np.vstack(points_list)
        mask = np.hstack(mask_list)

        # Randomly decide whether to apply augmentation
        if np.random.rand() < self.spherical_crop_prob:
            points, mask = self.apply_random_spherical_crop(points, mask)

        # Object rotation
        rotated_points = self.rotate_points(points)

        min_z_point = rotated_points[np.argmin(rotated_points[:, 2])]
        # Only calculate 2D xy plane distance
        xy_radius = np.max(np.linalg.norm(rotated_points[:, :2], axis=1))

        floor_z = min_z_point[2]
        object_height = np.max(rotated_points[:, 2]) - np.min(rotated_points[:, 2])

        # Randomly decide whether to add floor
        if np.random.rand() < self.floor_prob:
            floor_points = self.generate_floor(self.num_point, floor_z, xy_radius)
            floor_mask = np.zeros(floor_points.shape[0], dtype=np.int32)
            points = np.vstack([rotated_points, floor_points])
            mask = np.hstack([mask, floor_mask])

        # Randomly decide whether to add vertical plane
        if np.random.rand() < self.vertical_plane_prob:
            vertical_plane_points = self.generate_vertical_plane(self.num_point, floor_z, xy_radius, object_height, rotated_points)
            plane_mask = np.zeros(vertical_plane_points.shape[0], dtype=np.int32)
            points = np.vstack([points, vertical_plane_points])
            mask = np.hstack([mask, plane_mask])

        # Randomly decide whether to add horizontal plane (height between 0.8–1.0 object height, width between 60%–100% of radius)
        if np.random.rand() < self.horizontal_plane_prob:
            # Take random height factor
            h_factor = np.random.uniform(0.7, 0.9)
            plane_h = floor_z + h_factor * object_height

            # Generate plane and concatenate
            hp_pts = self.generate_horizontal_plane(self.num_point, plane_h, xy_radius)
            hp_mask = np.zeros(hp_pts.shape[0], dtype=np.int32)

            points = np.vstack([points, hp_pts])
            mask   = np.hstack([mask, hp_mask])

        # Randomly decide whether to add viewpoint point clouds from other objects (background)
        if np.random.rand() < self.other_object_prob and len(self.scene_folders) > 1:
            # Current object's center Z value
            center_z = np.mean(rotated_points[:, 2])

            # Find the direction of the farthest point
            xy_points = rotated_points[:, :2]
            distances = np.linalg.norm(xy_points, axis=1)
            max_distance_idx = np.argmax(distances)
            direction = xy_points[max_distance_idx]
            direction /= np.linalg.norm(direction) + 1e-6  # Normalize direction

            # Set offset in this direction
            offset_distance = np.random.uniform(1.0, 1.5) * xy_radius
            offset_xy = direction * offset_distance

            # Randomly select another scene and its viewpoint
            other_scene_names = [name for name in self.npz_files if name != scene_name]
            other_scene = np.random.choice(other_scene_names)
            other_files = self.npz_files[other_scene]
            other_file = np.random.choice(other_files)

            # Load and sample
            other_data = np.load(other_file)
            other_pts = other_data['p_w']
            N = other_pts.shape[0]
            if N == 0:
                print(f'ERROR no points in {other_file}, skipping this file')
                # Skip adding other object if file is empty
                pass
            else:
                idxs = np.random.choice(N, self.num_point, replace=(N < self.num_point))
                other_sampled = other_pts[idxs]

                # Z alignment - center height difference
                z_offset = center_z - np.mean(other_sampled[:, 2])
                other_sampled[:, 2] += z_offset

                # XY offset
                other_sampled[:, 0] += offset_xy[0]
                other_sampled[:, 1] += offset_xy[1]

                # Concatenate into background
                points = np.vstack([points, other_sampled])
                mask = np.hstack([mask, np.zeros(other_sampled.shape[0], dtype=np.int32)])


        # Randomly decide whether to generate outliers
        if np.random.rand() < self.outlier_prob:
            num_outliers = int(self.num_point * self.outlier_ratio)
            outliers = self.generate_outliers(num_outliers)
            points = np.vstack([points, outliers])
            mask = np.hstack([mask, np.zeros(num_outliers, dtype=np.int32)])

        # Before final sampling, apply bounding cylinder cropping
        points, mask = self.apply_bounding_cylinder_crop(points, mask, xy_radius)

        # Force apply scaling
        points = self.apply_random_scaling(points)

        # Finally, ensure point cloud count is 4096
        points, mask = self.sample_to_fixed_num(points, mask, self.num_point)

        # Shuffle point cloud data and mask
        permutation = np.random.permutation(points.shape[0])
        points = points[permutation]
        mask = mask[permutation]

        return points.astype(np.float32), mask.astype(np.int32)

    def generate_outliers(self, num_outliers):
        # Line type: points uniformly distributed along a line
        # Sphere type: points randomly distributed on sphere surface
        # Random points: completely randomly distributed points in space
        shape_type = np.random.choice(['line', 'sphere', 'point'])
        range_factor = 0.5
        if shape_type == 'line':
            out = np.random.uniform(-range_factor, range_factor, (num_outliers, 3))
            out[:, 0] = np.linspace(-range_factor, range_factor, num_outliers)
        elif shape_type == 'sphere':
            theta = np.random.uniform(0, 2 * np.pi, num_outliers)
            phi = np.random.uniform(0, np.pi, num_outliers)
            r = np.random.uniform(0.1, range_factor, num_outliers)
            out = np.vstack((
                r * np.sin(phi) * np.cos(theta),
                r * np.sin(phi) * np.sin(theta),
                r * np.cos(phi)
            )).T
        else:
            out = np.random.uniform(-range_factor, range_factor, (num_outliers, 3))
        return out

    def save_ply(self, filename, points):
        vertices = np.array([tuple(p) for p in points], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        vertex_element = PlyElement.describe(vertices, 'vertex')
        ply_data = PlyData([vertex_element], text=True)
        ply_data.write(filename)
        print(f"[INFO] Saved PLY file: {filename}")


class ShapeNetMultiClassDataset(Dataset):
    """
    Multi-class ShapeNet dataset for binary foreground/background segmentation.

    Expected layout:
    - data_root/<class_id>_dep/<object_id>/dep_pcl_*.npz
    - split_txt_dir/<class_id>/<mode>.lst
    """

    def __init__(self,
                 data_root,
                 mode='train',
                 split_txt_dir='split',
                 num_point=4096,
                 outlier_ratio=0.1,
                 outlier_prob=0,
                 floor_prob=1,
                 vertical_plane_prob=0.2,
                 spherical_crop_prob=0,
                 other_object_prob=1,
                 horizontal_plane_prob=0):
        super().__init__()

        self.data_root = pathlib.Path(data_root).expanduser()
        if not self.data_root.is_dir():
            raise FileNotFoundError(f"data_root not found: {self.data_root}")

        self.split_txt_dir = pathlib.Path(split_txt_dir).expanduser()
        if not self.split_txt_dir.is_dir():
            raise FileNotFoundError(f"split_txt_dir not found: {self.split_txt_dir}")

        self.num_point = num_point
        self.outlier_ratio = outlier_ratio
        self.outlier_prob = outlier_prob
        self.floor_prob = floor_prob
        self.vertical_plane_prob = vertical_plane_prob
        self.spherical_crop_prob = spherical_crop_prob
        self.other_object_prob = other_object_prob
        self.horizontal_plane_prob = horizontal_plane_prob

        self.class_names = ['02691156', '02933112', '03001627', '04090263', '04256520', '04401088']
        self.class_to_id = {name: idx for idx, name in enumerate(self.class_names)}

        self.object_data = []
        for class_name in self.class_names:
            class_id = self.class_to_id[class_name]
            split_path = self.split_txt_dir / class_name / f"{mode}.lst"
            if not split_path.is_file():
                print(f"Warning: Split file not found for class {class_name}: {split_path}")
                continue

            with open(split_path, 'r') as f:
                object_ids = [line.strip() for line in f if line.strip()]

            class_data_dir = self.data_root / f"{class_name}_dep"
            if not class_data_dir.is_dir():
                print(f"Warning: Class data directory not found: {class_data_dir}")
                continue

            for object_id in object_ids:
                object_path = class_data_dir / object_id
                if not object_path.is_dir():
                    continue
                npz_files = list(object_path.glob("dep_pcl_*.npz"))
                if len(npz_files) >= 3:
                    self.object_data.append((class_id, object_id, object_path))
                else:
                    print(f"Warning: Object {object_id} has fewer than 3 point cloud files")

        if not self.object_data:
            raise RuntimeError(f"No valid objects found for mode={mode}")

        print(f"[INFO] Mode: {mode} | Total samples: {len(self.object_data)}")
        print("[INFO] Class distribution:")
        for class_id, class_name in enumerate(self.class_names):
            count = sum(1 for cid, _, _ in self.object_data if cid == class_id)
            print(f"  {class_name}: {count} samples")

    def __len__(self):
        return len(self.object_data)

    def generate_horizontal_plane(self, num_points, height, max_radius):
        plane_size = np.random.uniform(0.8 * max_radius, 1.3 * max_radius)
        half = plane_size / 2

        x = np.random.uniform(-half, half, num_points)
        y = np.random.uniform(-half, half, num_points)
        z = np.full(num_points, height)
        pts = np.column_stack((x, y, z))

        theta = np.random.uniform(0, 2 * np.pi)
        c, s = np.cos(theta), np.sin(theta)
        rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        pts = pts @ rz.T
        return pts

    def generate_floor(self, num_points, floor_z, max_radius):
        scale_factor = np.random.uniform(3, 5)
        floor_size = max_radius * scale_factor
        x = np.random.uniform(-floor_size / 2, floor_size / 2, num_points)
        y = np.random.uniform(-floor_size / 2, floor_size / 2, num_points)
        z = np.full(num_points, floor_z)
        floor_points = np.column_stack((x, y, z))

        num_holes = np.random.randint(3, 7)
        for _ in range(num_holes):
            hole_radius = np.random.uniform(floor_size * 0.05, floor_size * 0.1)
            cx, cy = np.random.uniform(-floor_size / 2, floor_size / 2, 2)
            dists = np.sqrt((floor_points[:, 0] - cx) ** 2 + (floor_points[:, 1] - cy) ** 2)
            floor_points = floor_points[dists > hole_radius]
            if floor_points.shape[0] < 10:
                break
        return floor_points

    def generate_vertical_plane(self, num_points, floor_z, max_radius, object_height, all_points):
        xy_points = all_points[:, :2]
        distances = np.linalg.norm(xy_points, axis=1)
        max_distance_idx = np.argmax(distances)
        max_x, max_y = xy_points[max_distance_idx]

        width = np.random.uniform(2 * max_radius, 5 * max_radius)
        length = np.random.uniform(0.8 * object_height, 2 * object_height)
        offset = np.random.uniform(0.1, 0.3) * max_radius

        x = np.full(num_points, max_x + offset)
        y = np.random.uniform(max_y - width / 2, max_y + width / 2, num_points)
        z = np.random.uniform(floor_z, floor_z + length, num_points)
        return np.column_stack((x, y, z))

    def rotate_points(self, points):
        angle_z = np.random.uniform(-180, 180) * np.pi / 180
        angle_x = 0
        angle_y = 0

        rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                       [np.sin(angle_z), np.cos(angle_z), 0],
                       [0, 0, 1]])
        rx = np.array([[1, 0, 0],
                       [0, np.cos(angle_x), -np.sin(angle_x)],
                       [0, np.sin(angle_x), np.cos(angle_x)]])
        ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                       [0, 1, 0],
                       [-np.sin(angle_y), 0, np.cos(angle_y)]])
        rot = np.dot(rz, np.dot(rx, ry))
        return np.dot(points, rot.T)

    def apply_random_spherical_crop(self, points, mask, num_crop_min=2, num_crop_max=4):
        num_crop = np.random.randint(num_crop_min, num_crop_max + 1)
        for _ in range(num_crop):
            center = np.random.uniform(np.min(points, axis=0), np.max(points, axis=0), 3)
            radius = np.random.uniform(0.05, 0.2)
            distances = np.linalg.norm(points - center, axis=1)
            mask = mask[distances > radius]
            points = points[distances > radius]
        return points, mask

    def apply_bounding_cylinder_crop(self, points, mask, xy_radius):
        random_radius_factor = np.random.uniform(0.9, 1.1)
        final_radius = xy_radius * random_radius_factor
        xy_offset = np.random.uniform(-xy_radius * 0.2, xy_radius * 0.2, 2)
        center = np.array([xy_offset[0], xy_offset[1], 0])
        distances_to_center = np.linalg.norm(points[:, :2] - center[:2], axis=1)
        mask = mask[distances_to_center <= final_radius]
        points = points[distances_to_center <= final_radius]
        return points, mask

    def apply_random_scaling(self, points):
        scale_factor = np.random.uniform(0.9, 1.1)
        return points * scale_factor

    def check_intersection_interval(self, i1, i2):
        center_i1 = np.sum(i1, axis=0) / 2.0
        center_i2 = np.sum(i2, axis=0) / 2.0
        width_i1 = i1[1] - i1[0]
        width_i2 = i2[1] - i2[0]
        return np.all(abs(center_i1 - center_i2) < (width_i1 + width_i2) / 2)

    def check_bbox_overlap(self, bbox1, bbox2):
        return self.check_intersection_interval(bbox1, bbox2)

    def sample_to_fixed_num(self, points, mask, num_points=4096):
        if points.shape[0] > num_points:
            sampled_indices = np.random.choice(points.shape[0], num_points, replace=False)
            points = points[sampled_indices]
            mask = mask[sampled_indices]
        elif points.shape[0] < num_points:
            diff = num_points - points.shape[0]
            padding_points = np.zeros((diff, 3))
            padding_mask = np.zeros(diff, dtype=np.int32)
            points = np.vstack([points, padding_points])
            mask = np.hstack([mask, padding_mask])
        return points, mask

    def __getitem__(self, idx):
        class_id, _, object_path = self.object_data[idx]
        npz_files = sorted(object_path.glob("dep_pcl_*.npz"))

        num_views = np.random.randint(2, 5)
        selected_files = np.random.choice(npz_files, num_views, replace=False)

        points_list = []
        mask_list = []

        for f in selected_files:
            data = np.load(f)
            pts = data['p_w']
            n = pts.shape[0]
            idxs = np.random.choice(n, self.num_point, replace=(n < self.num_point))
            sampled = pts[idxs]
            points_list.append(sampled)
            mask_list.append(np.ones(self.num_point, dtype=np.int32))

        points = np.vstack(points_list)
        mask = np.hstack(mask_list)

        if np.random.rand() < self.spherical_crop_prob:
            points, mask = self.apply_random_spherical_crop(points, mask)

        rotated_points = self.rotate_points(points)
        floor_z = rotated_points[np.argmin(rotated_points[:, 2]), 2]
        xy_radius = np.max(np.linalg.norm(rotated_points[:, :2], axis=1))
        object_height = np.max(rotated_points[:, 2]) - np.min(rotated_points[:, 2])

        if np.random.rand() < self.floor_prob:
            floor_points = self.generate_floor(self.num_point, floor_z, xy_radius)
            floor_mask = np.zeros(floor_points.shape[0], dtype=np.int32)
            points = np.vstack([rotated_points, floor_points])
            mask = np.hstack([mask, floor_mask])

        if np.random.rand() < self.vertical_plane_prob:
            vertical_plane_points = self.generate_vertical_plane(
                self.num_point, floor_z, xy_radius, object_height, rotated_points
            )
            plane_mask = np.zeros(vertical_plane_points.shape[0], dtype=np.int32)
            points = np.vstack([points, vertical_plane_points])
            mask = np.hstack([mask, plane_mask])

        if np.random.rand() < self.horizontal_plane_prob:
            h_factor = np.random.uniform(0.7, 0.9)
            plane_h = floor_z + h_factor * object_height
            hp_pts = self.generate_horizontal_plane(self.num_point, plane_h, xy_radius)
            hp_mask = np.zeros(hp_pts.shape[0], dtype=np.int32)
            points = np.vstack([points, hp_pts])
            mask = np.hstack([mask, hp_mask])

        if np.random.rand() < self.other_object_prob and len(self.object_data) > 1:
            other_object_data = [data for data in self.object_data if data[0] != class_id]
            if other_object_data:
                other_idx = np.random.choice(len(other_object_data))
                _, _, other_object_path = other_object_data[other_idx]
                other_files = sorted(other_object_path.glob("dep_pcl_*.npz"))
                other_file = np.random.choice(other_files)

                other_data = np.load(other_file)
                other_pts = other_data['p_w']
                n = other_pts.shape[0]
                idxs = np.random.choice(n, self.num_point, replace=(n < self.num_point))
                other_sampled = other_pts[idxs]

                min_z_current = np.min(rotated_points[:, 2])
                min_z_other = np.min(other_sampled[:, 2])
                other_sampled[:, 2] += (min_z_current - min_z_other)

                current_bbox = np.array([
                    [np.min(rotated_points[:, 0]), np.min(rotated_points[:, 1])],
                    [np.max(rotated_points[:, 0]), np.max(rotated_points[:, 1])]
                ])

                success = False
                for _ in range(20):
                    theta = np.random.uniform(0, 2 * np.pi)
                    direction = np.array([np.cos(theta), np.sin(theta)])
                    current_size = np.max(current_bbox[1] - current_bbox[0])
                    offset_distance = np.random.uniform(current_size * 0.5, current_size * 1.2)
                    other_center = direction * offset_distance
                    other_bbox = np.array([
                        [np.min(other_sampled[:, 0]) + other_center[0], np.min(other_sampled[:, 1]) + other_center[1]],
                        [np.max(other_sampled[:, 0]) + other_center[0], np.max(other_sampled[:, 1]) + other_center[1]]
                    ])
                    if not self.check_bbox_overlap(current_bbox, other_bbox):
                        other_sampled[:, 0] += other_center[0]
                        other_sampled[:, 1] += other_center[1]
                        success = True
                        break

                if success:
                    points = np.vstack([points, other_sampled])
                    mask = np.hstack([mask, np.zeros(other_sampled.shape[0], dtype=np.int32)])

        if np.random.rand() < self.outlier_prob:
            num_outliers = int(self.num_point * self.outlier_ratio)
            outliers = self.generate_outliers(num_outliers)
            points = np.vstack([points, outliers])
            mask = np.hstack([mask, np.zeros(num_outliers, dtype=np.int32)])

        points, mask = self.apply_bounding_cylinder_crop(points, mask, xy_radius)
        points = self.apply_random_scaling(points)
        points, mask = self.sample_to_fixed_num(points, mask, self.num_point)

        permutation = np.random.permutation(points.shape[0])
        points = points[permutation]
        mask = mask[permutation]
        return points.astype(np.float32), mask.astype(np.int32)

    def generate_outliers(self, num_outliers):
        shape_type = np.random.choice(['line', 'sphere', 'point'])
        range_factor = 0.5
        if shape_type == 'line':
            out = np.random.uniform(-range_factor, range_factor, (num_outliers, 3))
            out[:, 0] = np.linspace(-range_factor, range_factor, num_outliers)
        elif shape_type == 'sphere':
            theta = np.random.uniform(0, 2 * np.pi, num_outliers)
            phi = np.random.uniform(0, np.pi, num_outliers)
            r = np.random.uniform(0.1, range_factor, num_outliers)
            out = np.vstack((
                r * np.sin(phi) * np.cos(theta),
                r * np.sin(phi) * np.sin(theta),
                r * np.cos(phi)
            )).T
        else:
            out = np.random.uniform(-range_factor, range_factor, (num_outliers, 3))
        return out

    def save_ply(self, filename, points):
        vertices = np.array([tuple(p) for p in points], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        vertex_element = PlyElement.describe(vertices, 'vertex')
        ply_data = PlyData([vertex_element], text=True)
        ply_data.write(filename)
        print(f"[INFO] Saved PLY file: {filename}")
