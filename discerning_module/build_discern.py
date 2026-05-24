"""Build segmentation (discerning) networks and load checkpoints.

Checkpoint paths must always be provided by the caller (e.g. training_scannet.build_seg_wrapper);
this module does not define default weight locations.
"""
import os

import torch


def _require_checkpoint_path(path, arg_name="checkpoint_path"):
    if path is None or (isinstance(path, str) and not str(path).strip()):
        raise ValueError(
            f"DiscerningModuleBuilder requires a non-empty {arg_name}; "
            f"defaults are not defined in discerning_module.build_seg."
        )
    resolved = os.path.abspath(os.path.expanduser(str(path).strip()))
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"DiscerningModule checkpoint file not found: {resolved!r}")
    return resolved


class DiscerningModuleBuilder:
    @staticmethod
    def sparse_unet(ckpt_path):
        path = _require_checkpoint_path(ckpt_path, "ckpt_path")
        from discerning_module.sparseUnet import SpUNetBase
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(path, map_location=device)
        model = SpUNetBase(
            in_channels=3,
            num_classes=2,
            base_channels=32).to(device)
        if 'model_state_dict' not in checkpoint:
            raise ValueError(
                f"Checkpoint at {path!r} has no 'model_state_dict' key."
            )
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'loading seg model from {path}')
        print("Model weights loaded successfully")
        model.train()
        print('finish building discerning module (sparseUnet)')
        return model

    @staticmethod
    def point_transformer(in_channels=3, num_classes=2, checkpoint_path=None):
        path = _require_checkpoint_path(checkpoint_path, "checkpoint_path")
        from discerning_module.pointTransformer import PointTransformerV2
        net = PointTransformerV2(in_channels=in_channels,
                                 num_classes=num_classes).cuda()
        checkpoint = torch.load(path, map_location="cuda")
        if 'model_state_dict' in checkpoint:
            net.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded PointTransformer checkpoint from '{path}' ")
        else:
            net.load_state_dict(checkpoint)
            print(f"Loaded PointTransformer state_dict from '{path}'")

        net.train()
        print('finish building discerning module (PointTransformer)')
        return net

    @staticmethod
    def point_net(checkpoint_path=None):
        path = _require_checkpoint_path(checkpoint_path, "checkpoint_path")
        from discerning_module.pointnet2_sem_seg import pointnet2
        print(f"build PointNet discerning module (ckpt={path})")
        net = pointnet2(num_classes=2, input_channel=3).cuda()
        ck = torch.load(path, map_location="cuda")
        net.load_state_dict(ck.get("state_dict", ck))
        print(f"Loaded PointNet checkpoint from '{path}'")
        net.train()
        print('finish building discerning module (PointNet)')
        return net
