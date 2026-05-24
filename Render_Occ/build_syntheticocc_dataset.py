import sys
import pathlib

root_dir = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

import os
import os.path as osp
import tqdm
import numpy as np
import trimesh
import random
import pickle
from scipy.spatial.transform import Rotation as R
import torch
from pointnet2.pointnet2 import furthest_point_sample
from lib.helper_ply import read_ply, write_ply
import colorsys
from typing import List, Tuple
import functools
import open3d as o3d
import glob

# ======================== Config (Edit Here) ========================
SRC_DIR = '/media/SSD/zihui/simon/data/shapenet_rendered'
SPLIT_DIR = '/media/SSD/zihui/simon/EvObj/data/shapenet_splits'
SAVE_DIR = '../data/synthetic_occ_5000/'
os.makedirs(SAVE_DIR, exist_ok=True)
VIS_DIR = os.path.join(SAVE_DIR, 'vis')
os.makedirs(VIS_DIR, exist_ok=True)
# ===================================================================

@functools.lru_cache(20)
def get_evenly_distributed_colors(count: int) -> List[Tuple[np.uint8, np.uint8, np.uint8]]:
    HSV_tuples = [(x / count, 1.0, 1.0) for x in range(count)]
    return list(map(lambda x: (np.array(colorsys.hsv_to_rgb(*x)) * 255).astype(np.uint8),HSV_tuples))

def fps_downsample(pc, n_sample_point=1024):
    """
    Downsample a point cloud with Furthest Point Sampling (FPS) and return indexes of sampled points.
    :param pc: (N, 3).
    :return:
        fps_idx: (N',).
    """
    pc = torch.from_numpy(pc).unsqueeze(0).cuda().contiguous()
    fps_idx = furthest_point_sample(pc, n_sample_point)
    fps_idx = fps_idx.cpu().numpy()[0]
    return fps_idx


# Fix random seed
np.random.seed(0)
random.seed(0)

# Object classes to use from ShapeNet
classes = ['02691156', '04401088', '02933112', '04090263', '03001627', '04256520']

instance_color = np.vstack(get_evenly_distributed_colors(8))

# Hyperparams for Objects, five types
n_objects = [8, 7, 6, 5, 4]  # Number of objects per room (regarded as individual types)
scale_intervals = [
    [0.25],
    [0.25],
    [0.25],
    [0.25],
    [0.25],
]  # Object scales for individual types
dataset_size = 1000  # Number of rooms per type
split_percentages = [.8, 0.2]
split_names = ['val', 'test']
n_rooms = [int(p * dataset_size) for p in split_percentages]
y_angle_range = [0., 360.]

# Hyperparams for Background
xz_ground_range = [0.6, 1.]
ground_thickness = 0.01
ground_height = -0.5
ground_level = ground_height + ground_thickness
wall_thickness = 0.01
wall_height_range = [0.35, 0.4]

# Hyperparams for sampling point cloud
n_sample_point = 100000
n_sample_point_fps = 20000


def get_class_models(cl, split_name):
    ''' Returns list of models for specific class and split.
    '''
    split_file = osp.join(SPLIT_DIR, cl, '%s.lst' % (split_name))
    with open(split_file, 'r') as f:
        model_files = f.read().split('\n')
        model_files = list(filter(lambda x: len(x) > 0, model_files))
    return model_files


def sample_models(model_dict, n_object):
    ''' Samples n_object from model_dict
    '''
    classes = [cl for cl in model_dict.keys()]
    classes.sort()

    out_list = []
    cl_list = []
    for n_object in range(n_object):
        cl = random.choice(classes)
        cl_list.append(cl)
        model = random.choice(model_dict[cl])
        out_list.append('%s/%s' % (cl, model))
    return out_list, cl_list


def sample_scales(n_object, type_id):
    '''Samples n_object scales in intervl scale_interval
    '''
    scale_interval = scale_intervals[type_id]
    # out_list = [scale_interval[0] + np.random.rand() * (scale_interval[1] - scale_interval[0]) for i in range(n_object)]
    out_list = [scale_interval[0] for i in range(n_object)]
    return out_list


def load_pointclouds(model_list, scale_list):
    """
    Loads pre-rendered multi-view point clouds and randomly selects 1-2 views per model.
    The loaded point clouds are z-axis up and converted to y-axis up for compatibility.
    Also loads the full mesh point cloud for proper ground placement.
    """
    out_list = []
    for model_idx, model in enumerate(model_list):
        cl, m = model.split('/')
        
        # Path to the rendered point cloud directory - strip any whitespace
        pcl_dir = osp.join(SRC_DIR, cl + '_dep', m.strip())
        pcl_dir = pcl_dir.strip()  # Remove any trailing whitespace

        
        # Find all available view files - try multiple methods
        glob_pattern = osp.join(pcl_dir, 'dep_pcl_*.npz')
        # print(f'glob_pattern: {glob_pattern}')
        available_views = glob.glob(glob_pattern)
        # print(f'glob found {len(available_views)} files')
        
        # Load full mesh point cloud for proper ground placement
        full_mesh_path = osp.join(pcl_dir, 'full_mesh_pcl.npz')
        full_mesh_points = None
        if osp.exists(full_mesh_path):
            try:
                full_data = np.load(full_mesh_path)
                full_mesh_points = full_data['p_w']  # Shape: (N, 3), z-axis up
                
                # Ensure full_mesh_points is 2D with 3 columns
                if full_mesh_points.ndim == 1:
                    if len(full_mesh_points) % 3 == 0:
                        full_mesh_points = full_mesh_points.reshape(-1, 3)
                    else:
                        print(f"Error: Cannot reshape full mesh {len(full_mesh_points)} points to (N, 3)")
                        full_mesh_points = None
                elif full_mesh_points.shape[1] != 3:
                    print(f"Warning: Full mesh point cloud has {full_mesh_points.shape[1]} columns, expected 3")
                    full_mesh_points = None
                else:
                    # Convert from z-axis up to y-axis up: [x, y, z] -> [x, z, y]
                    full_mesh_points = full_mesh_points[:, [0, 2, 1]]
                    # Apply scaling
                    full_mesh_points = full_mesh_points * scale_list[model_idx]
            except Exception as e:
                print(f"Warning: Failed to load full mesh for {cl}/{m}: {e}")
                full_mesh_points = None
        else:
            print(f"Warning: Full mesh point cloud not found for {cl}/{m} at {full_mesh_path}")

        if len(available_views) == 0:
            print(f"Warning: No point cloud views found for {cl}/{m} in {pcl_dir}")
            continue
            
        # Randomly select 1-2 views to simulate occlusion
        n_views = np.random.randint(2, 5) 
        selected_views = np.random.choice(available_views, size=min(n_views, len(available_views)), replace=False)
        
        # Load and concatenate selected views
        all_points = []
        for view_file in selected_views:
            data = np.load(view_file)
            points = data['p_w']  # Shape: (N, 3), z-axis up
            # print(f"Loaded point cloud shape: {points.shape} from {view_file}")
            
            # Ensure points are 2D with 3 columns
            if points.ndim == 1:
                if len(points) % 3 == 0:
                    points = points.reshape(-1, 3)
                else:
                    print(f"Error: Cannot reshape {len(points)} points to (N, 3)")
                    continue
            elif points.shape[1] != 3:
                print(f"Warning: Point cloud has {points.shape[1]} columns, expected 3")
                continue
                
            all_points.append(points)
        
        if len(all_points) > 0:
            combined_points = np.concatenate(all_points, axis=0)
            # print(f"After concatenation shape: {combined_points.shape}")
            
            # Ensure combined_points is (N, 3)
            if combined_points.ndim == 3:
                combined_points = combined_points.reshape(-1, combined_points.shape[-1])
                # print(f"After reshape: {combined_points.shape}")
            
            # Convert from z-axis up to y-axis up: [x, y, z] -> [x, z, y]
            combined_points = combined_points[:, [0, 2, 1]]
            # print(f"After coordinate transform: {combined_points.shape}")
            
            # Apply scaling (point clouds are already centered)
            combined_points = combined_points * scale_list[model_idx]
            # print(f"After scaling: {combined_points.shape}")
            
            # Ensure the result is always (N, 3)
            if combined_points.shape[1] != 3:
                # print(f"ERROR: Final shape is wrong: {combined_points.shape}")
                continue
            
            # Create a pseudo-mesh object to maintain compatibility
            pseudo_mesh = type('PseudoMesh', (), {})()
            pseudo_mesh.vertices = combined_points
            pseudo_mesh.area = len(combined_points) * 0.0001  # Fake area for point allocation
            
            # Store full mesh points for proper ground placement
            pseudo_mesh.full_mesh_points = full_mesh_points
            
            # Create bounds property for compatibility
            min_bounds = np.min(combined_points, axis=0)
            max_bounds = np.max(combined_points, axis=0)
            # print(f"min_bounds shape: {min_bounds.shape}, max_bounds shape: {max_bounds.shape}")
            pseudo_mesh.bounds = np.stack([min_bounds, max_bounds])
            # print(f"bounds shape: {pseudo_mesh.bounds.shape}")
            
            out_list.append(pseudo_mesh)
        
    return out_list


def sample_poses(mesh_list, y_angles):
    out_list = []
    pose_list = []
    for mesh_idx, mesh in enumerate(mesh_list):
        r = R.from_euler('y', y_angles[mesh_idx], degrees=True).as_matrix()
        mat = np.eye(4)
        mat[:3, :3] = r

        # Create a copy of the pseudo-mesh for point clouds
        out_mesh = type('PseudoMesh', (), {})()
        out_mesh.area = mesh.area
        
        # Apply rotation to point cloud
        vertices = np.array(mesh.vertices)
        rotated_vertices = vertices @ r.T
        
        # Use full mesh points for proper ground placement if available
        if not (hasattr(mesh, 'full_mesh_points') and mesh.full_mesh_points is not None):
            raise RuntimeError("full_mesh_points is required for ground placement but is missing.")
        min_y_full = np.min(mesh.full_mesh_points[:, 1])
        # Place point cloud onto the ground using full mesh minimum
        y_transl = ground_level - min_y_full
        
        rotated_vertices[:, 1] = rotated_vertices[:, 1] + y_transl
        mat[1, 3] = y_transl
        
        out_mesh.vertices = rotated_vertices
   
        # Update bounds after transformation
        min_bounds = np.min(rotated_vertices, axis=0)
        max_bounds = np.max(rotated_vertices, axis=0)
        out_mesh.bounds = np.stack([min_bounds, max_bounds])

        out_list.append(out_mesh)
        pose_list.append(mat)
    return out_list, pose_list


def draw_sample(bounds, it=0, method='uniform', dx=0.1, sigma=0.05, xz_range=[1., 1.]):
    ''' Draws a sample for provided method and given bounding box.
    '''
    if method == 'uniform':
        loc0 = -xz_range / 2. + wall_thickness
        loc_len = xz_range - bounds - 2 * wall_thickness
        loc = loc0 + np.random.rand(2) * loc_len
    if method == 'gaussian':
        mu_list = [[-0.5 + dx, -0.5 + dx],
                   [0.5 - dx, -0.5 + dx],
                   [-0.5 + dx, 0.5 - dx],
                   [0.5 - dx, 0.5 - dx],
                   [0., 0.],
                   ]
        while (True):
            loc = mu_list[it] + np.random.randn(2) * sigma
            if np.all(loc > -0.5) and np.all(loc + bounds < 0.5):
                break
    if method == 'uniform_structured':
        loc0 = [
            [-0.5, -0.5],
            [-0.5, 0.],
            [0., -0.5],
            [0., 0.],
        ]
        loc = loc0[it] + np.random.rand(2) * (0.5 - bounds)
    return loc


def check_intersection_interval(i1, i2):
    ''' Checks if the 2D intervals intersect.
    '''
    # i1, i2 of shape 2 x 2
    center_i1 = np.sum(i1, axis=0) / 2.
    center_i2 = np.sum(i2, axis=0) / 2.
    width_i1 = i1[1] - i1[0]
    width_i2 = i2[1] - i2[0]
    return np.all(abs(center_i1 - center_i2) < (width_i1 + width_i2) / 2)


def sample_locations(mesh_list, xz_range, poses, max_iter=1000):
    """
    Samples locations for the provided mesh list.
    """
    meshes = []
    bboxes = []
    poses_translated = []
    for mesh_idx, mesh in enumerate(mesh_list):
        try:
            # get bounds
            bounds = (mesh.bounds[1] - mesh.bounds[0])[[0, 2]]
            # sample location
            found_loc = False
            it = 0
            while (not found_loc):
                it += 1
                if it > max_iter:
                    raise ValueError("Maximum number of iterations exceeded!")
                loc0 = draw_sample(bounds, method='uniform', it=mesh_idx, xz_range=xz_range)
                bbox_i = np.array([loc0, loc0 + bounds])
                found_loc = True
                for bbox in bboxes:
                    if check_intersection_interval(bbox_i, bbox):
                        found_loc = False
                        break
            bboxes.append(bbox_i)

            # translate mesh with safe shape handling
            # Ensure mesh.vertices is 2D and extract x,z coordinates  
            vertices = np.array(mesh.vertices)
            # print(f"Initial vertices shape: {vertices.shape}")
            if vertices.ndim > 2:
                vertices = vertices.reshape(-1, vertices.shape[-1])
                # print(f"Reshaped vertices shape: {vertices.shape}")
            
            vertices_xz = vertices[:, [0, 2]]  # Shape: (N, 2)
            # print(f"vertices_xz shape: {vertices_xz.shape}")
            min_xz = np.min(vertices_xz, axis=0)    # Shape: (2,)
            # print(f"min_xz shape: {min_xz.shape}")
            
            # Ensure loc0 is 1D
            loc0 = np.array(loc0).flatten()  # Ensure shape (2,)
            # print(f"loc0 final shape: {loc0.shape}")
            xz_transl = loc0 - min_xz        # Shape: (2,)
            # print(f"xz_transl shape: {xz_transl.shape}")
            
            # Apply translation
            vertices_xz_new = vertices_xz + xz_transl.reshape(1, -1)
            # print(f"vertices_xz_new shape: {vertices_xz_new.shape}")
            
            # Update the mesh vertices
            vertices[:, [0, 2]] = vertices_xz_new
            mesh.vertices = vertices
            meshes.append(mesh)
            
            # translate pose
            pose = poses[mesh_idx].copy()
            pose[0, 3] = xz_transl[0]
            pose[2, 3] = xz_transl[1]
            poses_translated.append(pose)
            

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise e

    return meshes, poses_translated


def sample_pointcloud(cls_list, meshes, walls, ground, xz_range):
    """
    Process point cloud from pre-loaded point clouds and background meshes.
    """
    n_object = len(meshes)
    background_meshes = [ground] + walls

    # Calculate point allocation
    object_areas = [mesh.area for mesh in meshes]
    if background_meshes:
        bg_areas = [mesh.area for mesh in background_meshes]
        all_areas = object_areas + bg_areas
    else:
        all_areas = object_areas
    
    c_vol = np.array(all_areas)
    c_vol /= sum(c_vol)
    n_points = [int(c * n_sample_point) for c in c_vol]

    # Process point clouds
    points, segms, cates = [], [], []
    normals = []
    
    # Process object point clouds (pre-loaded)
    for i, mesh in enumerate(meshes):
        # mesh.vertices contains the pre-loaded point cloud
        pi = mesh.vertices
        
        # Subsample if we have more points than allocated
        if len(pi) > n_points[i] and n_points[i] > 0:
            indices = np.random.choice(len(pi), n_points[i], replace=False)
            pi = pi[indices]
        
        # Generate fake normals (since we don't have them from point clouds)
        norm_i = np.random.randn(len(pi), 3)
        norm_i = norm_i / np.linalg.norm(norm_i, axis=1, keepdims=True)
        
        # Foreground object has segment ids starting from 1
        segm = (i + 1) * np.ones(pi.shape[0], dtype=np.int16)
        cate = (int(classes.index(cls_list[i]))+1) * np.ones(pi.shape[0], dtype=np.int16)
        
        points.append(pi)
        normals.append(norm_i)
        segms.append(segm)
        cates.append(cate)
    
    # Process background meshes (walls and ground) - sample from mesh
    for i, mesh in enumerate(background_meshes):
        mesh_idx = n_object + i
        pi, face_idx = trimesh.sample.sample_surface_even(mesh, n_points[mesh_idx])
        norm_i = mesh.face_normals[face_idx]
        
        segm = np.zeros(pi.shape[0], dtype=np.int16)
        cate = np.zeros(pi.shape[0], dtype=np.int16)
        
        points.append(pi)
        normals.append(norm_i)
        segms.append(segm)
        cates.append(cate)
    points = np.concatenate(points, axis=0).astype(np.float32)
    normals = np.concatenate(normals, axis=0).astype(np.float32)
    segms = np.concatenate(segms, axis=0).astype(np.int16)
    cates = np.concatenate(cates, axis=0).astype(np.int16)

    # Remove the thickness of ground & wall from pointcloud
    mask = points[:, 1] > (ground_level - 1e-4)
    mask &= points[:, 2] > (- xz_range[1] / 2. + wall_thickness - 1e-4)
    mask &= points[:, 0] > (- xz_range[0] / 2. + wall_thickness - 1e-4)
    mask &= points[:, 2] < (+ xz_range[1] / 2. - wall_thickness + 1e-4)
    mask &= points[:, 0] < (+ xz_range[0] / 2. - wall_thickness + 1e-4)
    points = points[mask]
    normals = normals[mask]
    segms = segms[mask]
    cates = cates[mask]

    # FPS downsample
    fps_idx = fps_downsample(points, n_sample_point=n_sample_point_fps)
    points = points[fps_idx]
    normals = normals[fps_idx]
    segms = segms[fps_idx]
    cates = cates[fps_idx]
    return points, normals, segms, cates


def get_y_angles(n_object):
    angles = y_angle_range[0] + np.random.rand(n_object) * (y_angle_range[1] - y_angle_range[0])
    return angles


def get_walls(xz_range=[1., 1.], wall_height=0.2):
    out_list = []

    wall_x = trimesh.creation.box((xz_range[0], wall_height, wall_thickness))
    # put on ground plane and move to corner
    wall_x.vertices[:, 1] = wall_x.vertices[:, 1] - min(wall_x.vertices[:, 1]) + ground_level
    wall_x.vertices[:, 2] = wall_x.vertices[:, 2] - min(wall_x.vertices[:, 2]) - xz_range[1] / 2.
    out_list.append(wall_x)

    wall_x = trimesh.creation.box((wall_thickness, wall_height, xz_range[1]))
    # put on ground plane and move to corner
    wall_x.vertices[:, 1] = wall_x.vertices[:, 1] - min(wall_x.vertices[:, 1]) + ground_level
    wall_x.vertices[:, 0] = wall_x.vertices[:, 0] - min(wall_x.vertices[:, 0]) - xz_range[0] / 2.
    out_list.append(wall_x)

    wall_x = trimesh.creation.box((xz_range[0], wall_height, wall_thickness))
    # put on ground plane and move to corner
    wall_x.vertices[:, 1] = wall_x.vertices[:, 1] - min(wall_x.vertices[:, 1]) + ground_level
    wall_x.vertices[:, 2] = wall_x.vertices[:, 2] - max(wall_x.vertices[:, 2]) + xz_range[1] / 2.
    out_list.append(wall_x)

    wall_x = trimesh.creation.box((wall_thickness, wall_height, xz_range[1]))
    # put on ground plane and move to corner
    wall_x.vertices[:, 1] = wall_x.vertices[:, 1] - min(wall_x.vertices[:, 1]) + ground_level
    wall_x.vertices[:, 0] = wall_x.vertices[:, 0] - max(wall_x.vertices[:, 0]) + xz_range[0] / 2.
    out_list.append(wall_x)
    return out_list


def get_ground(xz_range=[1., 1.]):
    x_len, z_len = xz_range
    ground = trimesh.creation.box((x_len, ground_thickness, z_len))
    bounds = ground.bounds
    ground.vertices = ground.vertices - (bounds.sum(0) / 2).reshape(1, 3)  # center around origin
    ground.vertices[:, 1] = ground.vertices[:, 1] - min(ground.vertices[:, 1]) + ground_height
    return ground


# Main loop of data generation
split_lsts = {'val': '', 'test': ''}
for type_id, n_object in enumerate(n_objects):
    room_id = 0
    pbar = tqdm.tqdm(total=sum(n_rooms))

    for split_id, split_name in enumerate(split_names):
        # Load objects of all categories under current split (train/val/test)
        model_files = {}
        for cl in classes:
            model_files[cl] = get_class_models(cl, split_name)
        # Loop over items for current split
        split_item_id = 0

        while split_item_id < n_rooms[split_id]:
            # Create meta info for the scene
            item_dict = {}
            item_dict['room_id'] = room_id
            item_dict['split'] = split_name
            item_dict['n_object'] = n_object
            obj_list, cl_list = sample_models(model_files, n_object)
            item_dict['objects'] = obj_list
            item_dict['classes'] = cl_list
            item_dict['scales'] = sample_scales(n_object, type_id)
            axis0 = np.random.rand() > 0.5  # 0 is x-axis, 1 is z-axis
            scale_axis = np.random.rand() * (xz_ground_range[1] - xz_ground_range[0]) + xz_ground_range[0]
            ranges = [1., scale_axis] if axis0 else [scale_axis, 1.]
            item_dict['xz_ground_range'] = np.array(ranges)
            item_dict['wall_height'] = wall_height_range[0] + np.random.rand() * (
                        wall_height_range[1] - wall_height_range[0])

            # Generate the 1st static frame
            canonical_meshes = load_pointclouds(item_dict['objects'], item_dict['scales'])
            init_y_angles = get_y_angles(n_object)
            meshes, poses = sample_poses(canonical_meshes, init_y_angles)
            try:
                meshes, poses = sample_locations(meshes, item_dict['xz_ground_range'], poses)
            except Exception as e:
                print('Error: ', e)
                continue

            # Generate background
            walls = get_walls(xz_range=item_dict['xz_ground_range'], wall_height=item_dict['wall_height'])
            ground = get_ground(xz_range=item_dict['xz_ground_range'])

            # Save path for the scene
            sample_name = '%02d_%06d' % (n_object, room_id)

            points, normals, segms, cates = sample_pointcloud(item_dict['classes'], meshes, walls, ground, xz_range=item_dict['xz_ground_range'])
            points, normals = 4*points[:, [0, 2, 1]], normals[:, [0, 2, 1]]

            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(points)
            # pcd.normals = o3d.utility.Vector3dVector(normals)
            #
            # # Visualize the point cloud with normals
            # o3d.visualization.draw_geometries([pcd], point_show_normal=True)


            os.makedirs(os.path.join(SAVE_DIR, split_name), exist_ok=True)
            os.makedirs(os.path.join(VIS_DIR, split_name), exist_ok=True)

            mask_ids = np.unique(segms[segms!=0])
            gtcolor = np.ones_like(points[:, 0:3]) * 128
            for mask_id in mask_ids:
                gtcolor[segms==mask_id] = instance_color[mask_id-1]
            
            ply_path = os.path.join(VIS_DIR, split_name, sample_name + '.ply')
            print(f"Saving PLY file: {ply_path}")
            write_ply(ply_path, [points[:, 0:3], gtcolor.astype(np.uint8)],
                      ['x', 'y', 'z', 'red', 'green', 'blue'])
            print(f"PLY file saved successfully with {len(points)} points")

            ### make it to be ScanNet format, semantic*1000 + instance_id
            segms = segms + cates*1000
            ###
            points = np.concatenate((points, normals, cates[:, None], segms[:, None]), -1)

            np.savez(os.path.join(SAVE_DIR, split_name, sample_name + '.npz'), data=points)



            if split_lsts[split_name] != '':
                split_lsts[split_name] += '\n'
            split_lsts[split_name] += sample_name

            room_id += 1
            split_item_id += 1
            pbar.update(1)

    pbar.close()

# Save the split info
for split_name in split_names:
    with open(osp.join(SAVE_DIR, split_name + '.lst'), 'w') as f:
        f.write(split_lsts[split_name])


os.system(f"mv {os.path.join(SAVE_DIR, 'val')} {os.path.join(SAVE_DIR, 'train')}")
os.system(f"mv {os.path.join(SAVE_DIR, 'val.lst')} {os.path.join(SAVE_DIR, 'train.lst')}")
