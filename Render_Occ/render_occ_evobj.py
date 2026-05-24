import os
import torch
import numpy as np
from tqdm import tqdm
from depth_render import DepthRender
from pytorch3d.structures import Meshes, join_meshes_as_batch
import trimesh
from glob import glob
import sys
import argparse
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def header_properties(field_list, field_names):

    # List of lines to write
    lines = []

    # First line describing element vertex
    lines.append('element vertex %d' % field_list[0].shape[0])

    # Properties lines
    i = 0
    for fields in field_list:
        for field in fields.T:
            lines.append('property %s %s' % (field.dtype.name, field_names[i]))
            i += 1

    return lines
def write_ply(filename, field_list, field_names, triangular_faces=None):
    """
    Write ".ply" files

    Parameters
    ----------
    filename : string
        the name of the file to which the data is saved. A '.ply' extension will be appended to the
        file name if it does no already have one.

    field_list : list, tuple, numpy array
        the fields to be saved in the ply file. Either a numpy array, a list of numpy arrays or a
        tuple of numpy arrays. Each 1D numpy array and each column of 2D numpy arrays are considered
        as one field.

    field_names : list
        the name of each fields as a list of strings. Has to be the same length as the number of
        fields.

    Examples
    --------
    >>> points = np.random.rand(10, 3)
    >>> write_ply('example1.ply', points, ['x', 'y', 'z'])

    >>> values = np.random.randint(2, size=10)
    >>> write_ply('example2.ply', [points, values], ['x', 'y', 'z', 'values'])

    >>> colors = np.random.randint(255, size=(10,3), dtype=np.uint8)
    >>> field_names = ['x', 'y', 'z', 'red', 'green', 'blue', 'values']
    >>> write_ply('example3.ply', [points, colors, values], field_names)

    """

    # Format list input to the right form
    field_list = list(field_list) if (type(field_list) == list or type(field_list) == tuple) else list((field_list,))
    for i, field in enumerate(field_list):
        if field.ndim < 2:
            field_list[i] = field.reshape(-1, 1)
        if field.ndim > 2:
            print('fields have more than 2 dimensions')
            return False

    # check all fields have the same number of data
    n_points = [field.shape[0] for field in field_list]
    if not np.all(np.equal(n_points, n_points[0])):
        print('wrong field dimensions')
        return False

    # Check if field_names and field_list have same nb of column
    n_fields = np.sum([field.shape[1] for field in field_list])
    if (n_fields != len(field_names)):
        print('wrong number of field names')
        return False

    # Add extension if not there
    if not filename.endswith('.ply'):
        filename += '.ply'

    # open in text mode to write the header
    with open(filename, 'w') as plyfile:

        # First magical word
        header = ['ply']

        # Encoding format
        header.append('format binary_' + sys.byteorder + '_endian 1.0')

        # Points properties description
        header.extend(header_properties(field_list, field_names))

        # Add faces if needded
        if triangular_faces is not None:
            header.append('element face {:d}'.format(triangular_faces.shape[0]))
            header.append('property list uchar int vertex_indices')

        # End of header
        header.append('end_header')

        # Write all lines
        for line in header:
            plyfile.write("%s\n" % line)

    # open in binary/append to use tofile
    with open(filename, 'ab') as plyfile:

        # Create a structured array
        i = 0
        type_list = []
        for fields in field_list:
            for field in fields.T:
                type_list += [(field_names[i], field.dtype.str)]
                i += 1
        data = np.empty(field_list[0].shape[0], dtype=type_list)
        i = 0
        for fields in field_list:
            for field in fields.T:
                data[field_names[i]] = field
                i += 1

        data.tofile(plyfile)

        if triangular_faces is not None:
            triangular_faces = triangular_faces.astype(np.int32)
            type_list = [('k', 'uint8')] + [(str(ind), 'int32') for ind in range(3)]
            data = np.empty(triangular_faces.shape[0], dtype=type_list)
            data['k'] = np.full((triangular_faces.shape[0],), 3, dtype=np.uint8)
            data['0'] = triangular_faces[:, 0]
            data['1'] = triangular_faces[:, 1]
            data['2'] = triangular_faces[:, 2]
            data.tofile(plyfile)

    return True



def generate_depth_point_cloud(mesh_path, save_path):
    # Load the mesh
    cls = mesh_path.split('/')[-3]  
    name = mesh_path.split('/')[-1][0:-4]
    print('###### Start File', name)
    mesh = trimesh.load(mesh_path)
    bbox = mesh.bounding_box.bounds
    loc = (bbox[0] + bbox[1]) / 2
    scale = (bbox[1] - bbox[0]).max()

    mesh.apply_translation(-loc)
    mesh.apply_scale(1.0 / scale)
    # mesh.vertices = mesh.vertices[:, [0, 2, 1]]  # Convert to xzy
    mesh.vertices = mesh.vertices[:, [0, 1, 2]]  

    # Convert to PyTorch3D mesh
    verts = torch.tensor(mesh.vertices, dtype=torch.float32)
    faces = torch.tensor(mesh.faces, dtype=torch.int64)
    pytorch3d_mesh = Meshes(verts=[verts.to(device)], faces=[faces.to(device)])
    mesh_list = []

    # Generate camera positions on the upper hemisphere
    radius = 2
    render_num = 12
    dist = np.random.uniform(radius, radius, size=render_num)
    # elev = np.linspace(-30, 30, render_num, endpoint=False)### rotation above xy plane
    # azim = np.linspace(80, 360, render_num, endpoint=False) ### rotation in xy plane
    # elev = 30*(np.random.rand(render_num)*2-1)### rotation above xy plane
    # azim = 180*(np.random.rand(render_num)*2-1) ### rotation in xy plane
    elev = 30 * (np.random.rand(render_num))  
    azim = 180*(np.random.rand(render_num)*2-1) ### rotation in xy plane
    for _ in range(render_num):
        mesh_list.append(pytorch3d_mesh)

    depth_renderer = DepthRender(dist, elev, azim, device)
    depth_list, R_list, t_list, coords_CAM_list, coords_OBJ_list = depth_renderer.render(
        join_meshes_as_batch(mesh_list))
    os.makedirs(os.path.join(save_path, cls + '_dep', name), exist_ok=True)
    

    for idx, point_cloud in enumerate(coords_OBJ_list):
        if len(point_cloud) > 5000:
            point_cloud = point_cloud[np.random.choice(len(point_cloud), 5000)]
        np.savez_compressed(os.path.join(save_path, cls + '_dep', name, 'dep_pcl_' + str(idx) + '.npz'),
                            p_w=point_cloud[:, [0, 2, 1]])
        # write_ply(os.path.join(save_path, cls + '_dep', name, 'dep_pcl_' + str(idx) + '.ply'),
        #           [point_cloud[:, [0, 2, 1]]], ['x', 'y', 'z'])
    
    sampled_points = mesh.sample(2048)

    sampled_points_transformed = sampled_points[:, [0, 2, 1]]
    

    np.savez_compressed(os.path.join(save_path, cls + '_dep', name, 'full_mesh_pcl.npz'),
                        p_w=sampled_points_transformed)
    # write_ply(os.path.join(save_path, cls + '_dep', name, 'full_mesh_pcl.ply'),
    #           [sampled_points_transformed], ['x', 'y', 'z'])

def main():
    parser = argparse.ArgumentParser(description="Render depth point clouds for ShapeNet meshes.")
    parser.add_argument(
        "--mesh-dir",
        default="/media/SSD/zihui/simon/data/shapenet",
        help="Root directory containing class subdirectories.",
    )
    parser.add_argument(
        "--save-dir",
        default="/media/SSD/zihui/simon/data/shapenet_rendered",
        help="Output directory for rendered point clouds.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["03001627"],
        help="Class IDs to process.",
    )
    # ['03001627', '04256520', '04401088', '02691156', '04090263', '02933112']
    # chair, sofa, telephone, airplane, rifle, cabinet
    args = parser.parse_args()

    for cls in args.classes:
        print("###### Start cls", cls)
        cls_dir = os.path.join(args.mesh_dir, cls)
        cls_meshes = sorted(glob(os.path.join(cls_dir, "4_watertight_scaled", "*.off")))
        total_meshes = len(cls_meshes)
        for idx, mesh_path in enumerate(tqdm(cls_meshes, desc=f"{cls} meshes", unit="mesh")):
            print(f"Processing {idx + 1}/{total_meshes} in class {cls}")
            generate_depth_point_cloud(mesh_path, args.save_dir)
            print(f"finished {mesh_path}")


if __name__ == "__main__":
    main()
