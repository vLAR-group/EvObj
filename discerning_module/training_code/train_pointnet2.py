import os
import sys
import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Local imports
from shapenet_dataset import ShapeNetDataset, ShapeNetMultiClassDataset

TRAINING_CODE_DIR = Path(__file__).resolve().parent
DISCERNING_MODULE_DIR = TRAINING_CODE_DIR.parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(DISCERNING_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(DISCERNING_MODULE_DIR))

from discerning_module.pointnet2_sem_seg import pointnet2


def seed_everything(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args():
    parser = argparse.ArgumentParser(description='Train PointNet++ on ShapeNet')
    default_data_root = '/media/SSD/zihui/simon/segnet_spconv/data/shapenet/4_watertight_scaled_dep'
    default_split_txt_dir = '/media/SSD/zihui/simon/segnet_spconv/data/shapenet'
    default_log_dir = PROJECT_ROOT / 'discerning_module' / 'training_code' / 'logs' / 'shapenet_pointnet2'

    parser.add_argument('--data_root', type=str,
                        default=str(default_data_root),
                        help='Path to shapenet data root')
    parser.add_argument('--split_txt_dir', type=str,
                        default=str(default_split_txt_dir),
                        help='Path to split files (single: train/test.txt, multicls: <class>/<mode>.lst)')
    parser.add_argument('--dataset_mode', type=str, default='single',
                        choices=['single', 'multicls'],
                        help='ShapeNet dataset mode: single class or 6-class multicls')
    parser.add_argument('--log_dir', type=str, default=str(default_log_dir),
                        help='Directory to save logs and checkpoints')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--num_point', type=int, default=4096,
                        help='Number of points per sample')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU id to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save_every', type=int, default=1,
                        help='Save model every N epochs')

    return parser.parse_args()


def prepare_pointnet2_input(points, masks, in_channels, device='cuda'):
    """
    Prepare input data for PointNet++ model.

    Args:
        points: [batch_size, num_points, C] torch tensor
        masks: [batch_size, num_points] torch tensor
        in_channels: number of input feature channels used by model
        device: device to place tensors on

    Returns:
        dict containing:
            - model_input: [batch_size, in_channels, num_points]
            - labels: [batch_size, num_points]
            - original_points: [batch_size, num_points, 3]
    """
    if points.shape[-1] < in_channels:
        raise ValueError(
            f'Input points have {points.shape[-1]} channels, but in_channels={in_channels}. '
            'Please reduce in_channels or provide matching features.'
        )

    features = points[..., :in_channels].to(device).float()  # [B, N, C]
    xyz = points[..., :3].to(device).float()  # [B, N, 3]
    labels = masks.to(device).long()  # [B, N]

    # Center xyz channels for training stability.
    if in_channels >= 3:
        xyz_centered = xyz - xyz.mean(dim=1, keepdim=True)
        features[..., :3] = xyz_centered

    model_input = features.transpose(1, 2).contiguous()  # [B, C, N]

    return {
        'model_input': model_input,
        'labels': labels,
        'original_points': xyz,
    }


def calculate_point_accuracy(pred_logits, labels):
    """
    Calculate accuracy on points.

    Args:
        pred_logits: [B, N, num_classes] model predictions
        labels: [B, N] point labels

    Returns:
        accuracy: float, accuracy on points
        point_predictions: list[tensor], predictions for each batch sample
        point_labels: list[tensor], labels for each batch sample
    """
    predictions = torch.argmax(pred_logits, dim=2)  # [B, N]

    total_correct = (predictions == labels).sum().item()
    total_points = labels.numel()
    accuracy = total_correct / total_points if total_points > 0 else 0.0

    point_predictions = [predictions[b] for b in range(predictions.shape[0])]
    point_labels = [labels[b] for b in range(labels.shape[0])]

    return accuracy, point_predictions, point_labels


def train_epoch(model, dataloader, criterion, optimizer, device, in_channels, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_point_correct = 0
    total_point_samples = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for points, masks in pbar:
        points = points.to(device).float()
        masks = masks.to(device).long()

        point_data = prepare_pointnet2_input(points, masks, in_channels, device)

        optimizer.zero_grad()

        logits, _ = model(point_data['model_input'])  # [B, N, num_classes]
        loss = criterion(logits.reshape(-1, logits.shape[-1]), point_data['labels'].reshape(-1))

        loss.backward()
        optimizer.step()

        point_accuracy, point_predictions, point_labels = calculate_point_accuracy(logits, point_data['labels'])

        total_loss += loss.item()
        for batch_point_preds, batch_point_labels in zip(point_predictions, point_labels):
            total_point_correct += (batch_point_preds == batch_point_labels).sum().item()
            total_point_samples += batch_point_labels.size(0)

        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'PointAcc': f'{point_accuracy:.4f}'
        })

    avg_loss = total_loss / len(dataloader)
    avg_point_accuracy = total_point_correct / total_point_samples

    return avg_loss, avg_point_accuracy


def save_ply(filename, points, colors=None):
    """Save point cloud as PLY file"""
    from plyfile import PlyData, PlyElement

    if colors is not None:
        vertices = np.array([tuple(list(p) + list(c)) for p, c in zip(points, colors)],
                            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                   ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    else:
        vertices = np.array([tuple(p) for p in points],
                            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

    vertex_element = PlyElement.describe(vertices, 'vertex')
    ply_data = PlyData([vertex_element], text=True)
    ply_data.write(filename)


def save_enhanced_validation_visualization(point_predictions, point_labels, original_points, vis_dir, epoch, sample_idx):
    """
    Save enhanced visualization PLY files for validation results.

    Args:
        point_predictions: [num_points] model predictions for all points of one sample
        point_labels: [num_points] original point labels
        original_points: [num_points, 3] original point coordinates
        vis_dir: directory to save PLY files
        epoch: current epoch number
        sample_idx: sample index for naming
    """
    point_preds = point_predictions.cpu().numpy()
    point_labels_np = point_labels.cpu().numpy()
    original_points_np = original_points.cpu().numpy() if torch.is_tensor(original_points) else original_points

    sample_name = f"epoch_{epoch:03d}_sample_{sample_idx:02d}"

    if len(original_points_np) > 0:
        point_input_colors = np.full((len(original_points_np), 3), [128, 128, 128], dtype=np.uint8)
        save_ply(vis_dir / f"{sample_name}_points_input.ply", original_points_np, point_input_colors)

        point_pred_fg_indices = np.where(point_preds == 1)[0]
        if len(point_pred_fg_indices) > 0:
            pred_fg_points = original_points_np[point_pred_fg_indices]
            pred_fg_point_colors = np.full((len(point_pred_fg_indices), 3), [0, 255, 0], dtype=np.uint8)
            save_ply(vis_dir / f"{sample_name}_points_pred_fg.ply", pred_fg_points, pred_fg_point_colors)
        else:
            save_ply(vis_dir / f"{sample_name}_points_pred_fg.ply", np.empty((0, 3)))

        point_gt_fg_indices = np.where(point_labels_np == 1)[0]
        if len(point_gt_fg_indices) > 0:
            gt_fg_points = original_points_np[point_gt_fg_indices]
            gt_fg_point_colors = np.full((len(point_gt_fg_indices), 3), [255, 0, 0], dtype=np.uint8)
            save_ply(vis_dir / f"{sample_name}_points_gt_fg.ply", gt_fg_points, gt_fg_point_colors)
        else:
            save_ply(vis_dir / f"{sample_name}_points_gt_fg.ply", np.empty((0, 3)))
    else:
        print(f"Warning: no original points for sample {sample_idx}")
        save_ply(vis_dir / f"{sample_name}_points_input.ply", np.empty((0, 3)))


def validate_epoch(model, dataloader, criterion, device, in_channels, save_vis=False, vis_dir=None, epoch=None):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    total_point_correct = 0
    total_point_samples = 0

    vis_preds = []
    vis_labels = []
    vis_points = []
    max_vis_samples = 10

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for points, masks in pbar:
            points = points.to(device).float()
            masks = masks.to(device).long()

            point_data = prepare_pointnet2_input(points, masks, in_channels, device)

            logits, _ = model(point_data['model_input'])
            loss = criterion(logits.reshape(-1, logits.shape[-1]), point_data['labels'].reshape(-1))
            point_accuracy, point_predictions, point_labels = calculate_point_accuracy(logits, point_data['labels'])

            total_loss += loss.item()
            for b, (batch_point_preds, batch_point_labels) in enumerate(zip(point_predictions, point_labels)):
                total_point_correct += (batch_point_preds == batch_point_labels).sum().item()
                total_point_samples += batch_point_labels.size(0)

                if save_vis and len(vis_preds) < max_vis_samples:
                    vis_preds.append(batch_point_preds.detach().cpu())
                    vis_labels.append(batch_point_labels.detach().cpu())
                    vis_points.append(point_data['original_points'][b].detach().cpu())

            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'PointAcc': f'{point_accuracy:.4f}'
            })

    if save_vis and vis_dir is not None and epoch is not None:
        for sample_idx, (preds, labels, xyz) in enumerate(zip(vis_preds, vis_labels, vis_points)):
            save_enhanced_validation_visualization(preds, labels, xyz, vis_dir, epoch, sample_idx)

    avg_loss = total_loss / len(dataloader)
    avg_point_accuracy = total_point_correct / total_point_samples

    return avg_loss, avg_point_accuracy


def setup_logging(log_dir):
    """Setup logging configuration"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_dir / 'train.log')
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def main():
    args = get_args()

    seed_everything(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger = setup_logging(args.log_dir)
    logger.info(f"Arguments: {vars(args)}")
    logger.info(f"Using device: {device}")
    logger.info(
        "Fixed settings: in_channels=3, num_classes=2, val_every=1"
    )

    logger.info("Creating datasets...")
    dataset_cls = ShapeNetDataset if args.dataset_mode == 'single' else ShapeNetMultiClassDataset
    logger.info(f"ShapeNet dataset mode: {args.dataset_mode} ({dataset_cls.__name__})")

    train_dataset = dataset_cls(
        mode='train',
        data_root=args.data_root,
        split_txt_dir=args.split_txt_dir,
        num_point=args.num_point,
    )

    val_dataset = dataset_cls(
        mode='test',
        data_root=args.data_root,
        split_txt_dir=args.split_txt_dir,
        num_point=args.num_point,
    )

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        drop_last=False
    )

    logger.info("Creating PointNet++ model...")
    model = pointnet2(
        num_classes=2,
        input_channel=3,
    ).to(device)
    logger.info(f"Model created successfully \n {model}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.NLLLoss()

    checkpoint_dir = Path(args.log_dir) / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    shapenet_vis_dir = Path(args.log_dir) / 'shapenet_val_vis'
    shapenet_vis_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting training...")
    best_val_point_acc = 0.0

    shapenet_train_point_accs = []
    shapenet_val_point_accs = []
    shapenet_train_losses = []
    shapenet_val_losses = []

    val_epochs = []

    for epoch in range(1, args.epochs + 1):
        logger.info(f"Epoch {epoch}/{args.epochs}")

        train_loss, train_point_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, 3, epoch
        )

        shapenet_train_point_accs.append(train_point_acc)
        shapenet_train_losses.append(train_loss)

        logger.info(f"Train Loss: {train_loss:.4f}, Train PointAcc: {train_point_acc:.4f}")

        if epoch % 1 == 0:
            val_loss, val_point_acc = validate_epoch(
                model, val_loader, criterion, device,
                3, save_vis=False, vis_dir=shapenet_vis_dir, epoch=epoch
            )

            shapenet_val_point_accs.append(val_point_acc)
            shapenet_val_losses.append(val_loss)
            val_epochs.append(epoch)

            logger.info(f"ShapeNet Val Loss: {val_loss:.4f}, Val PointAcc: {val_point_acc:.4f}")
            logger.info(f"ShapeNet validation visualization files saved to {shapenet_vis_dir}")

            if val_point_acc > best_val_point_acc:
                best_val_point_acc = val_point_acc
                best_checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'train_point_acc': train_point_acc,
                    'val_loss': val_loss,
                    'val_point_acc': val_point_acc,
                    'args': vars(args)
                }
                torch.save(best_checkpoint, checkpoint_dir / 'best_model.pth')
                logger.info(f"New best model saved with val_point_acc: {val_point_acc:.4f}")

        if epoch % args.save_every == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_point_acc': train_point_acc,
                'args': vars(args)
            }
            torch.save(checkpoint, checkpoint_dir / f'epoch_{epoch:03d}.pth')
            logger.info(f"Checkpoint saved for epoch {epoch}")

    logger.info(f"Training completed! Best validation point accuracy: {best_val_point_acc:.4f}")

    plot_training_curves(
        shapenet_train_point_accs, shapenet_val_point_accs,
        shapenet_train_losses, shapenet_val_losses,
        val_epochs, args.log_dir
    )


def plot_training_curves(shapenet_train_point_accs, shapenet_val_point_accs,
                         shapenet_train_losses, shapenet_val_losses,
                         val_epochs, log_dir):
    """
    Plot training curves for point-level accuracy and loss

    Args:
        shapenet_train_point_accs: list of ShapeNet training point accuracies
        shapenet_val_point_accs: list of ShapeNet validation point accuracies
        shapenet_train_losses: list of ShapeNet training losses
        shapenet_val_losses: list of ShapeNet validation losses
        val_epochs: list of epochs where validation was performed
        log_dir: directory to save plots
    """
    epochs = list(range(1, len(shapenet_train_point_accs) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.plot(epochs, shapenet_train_point_accs, 'b-', label='ShapeNet Train', linewidth=2, marker='o')
    ax1.plot(val_epochs, shapenet_val_point_accs, 'r-', label='ShapeNet Val', linewidth=2, marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Point-Level Accuracy')
    ax1.set_title('Point-Level Accuracy Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    ax2.plot(epochs, shapenet_train_losses, 'b-', label='ShapeNet Train', linewidth=2, marker='o')
    ax2.plot(val_epochs, shapenet_val_losses, 'r-', label='ShapeNet Val', linewidth=2, marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss Curves')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = Path(log_dir) / 'training_curves.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Training curves saved to: {plot_path}")


if __name__ == '__main__':
    main()
