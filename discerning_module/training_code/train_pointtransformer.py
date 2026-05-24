import os
import sys
import argparse
import logging
import random
import time
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
if str(DISCERNING_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(DISCERNING_MODULE_DIR))

from pointTransformer import PointTransformerV2

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
    parser = argparse.ArgumentParser(description='Train Point Transformer V2 on ShapeNet')
    default_data_root = '/media/SSD/zihui/simon/segnet_spconv/data/shapenet/4_watertight_scaled_dep'
    default_split_txt_dir = '/media/SSD/zihui/simon/segnet_spconv/data/shapenet'
    default_log_dir = PROJECT_ROOT / 'discerning_module' / 'training_code' / 'logs' / 'shapenet_spconv'

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


def prepare_point_transformer_input(points, masks, device='cuda'):
    """
    Prepare input data for Point Transformer V2 model
    
    Args:
        points: [batch_size, num_points, 3] torch tensor
        masks: [batch_size, num_points] torch tensor
        device: device to place tensors on

    Returns:
        dict containing:
            - coord: [total_points, 3] point coordinates (centered)
            - feat: [total_points, 3] features (centered point coordinates)
            - offset: [batch_size] cumulative point counts
            - labels: [total_points] point labels
            - original_points: [batch_size, num_points, 3] original point coordinates
            - original_masks: [batch_size, num_points] original point labels
    """
    batch_size = points.shape[0]
    
    all_coords = []
    all_features = []
    all_labels = []
    batch_offsets = [0]

    for b in range(batch_size):
        # Get coordinates and features
        coords = points[b].to(device).float()  # [num_points, 3]
        # Center coordinates and features by subtracting mean
        coords_centered = coords - coords.mean(dim=0, keepdim=True)  # [num_points, 3]
        features = coords_centered  # Use centered coordinates as features
        labels = masks[b].to(device).long()  # [num_points]
        
        all_coords.append(coords_centered)
        all_features.append(features)
        all_labels.append(labels)
        batch_offsets.append(batch_offsets[-1] + coords.shape[0])

    # Concatenate all batches
    coord = torch.cat(all_coords, dim=0)  # [total_points, 3]
    feat = torch.cat(all_features, dim=0)  # [total_points, 3]
    labels = torch.cat(all_labels, dim=0)  # [total_points]
    offset = torch.tensor(batch_offsets[1:], dtype=torch.long, device=device)  # [batch_size]

    return {
        'coord': coord,
        'feat': feat,
        'offset': offset,
        'labels': labels,
        'original_points': points,
        'original_masks': masks
    }


def calculate_point_accuracy(pred_logits, point_data):
    """
    Calculate accuracy on points
    
    Args:
        pred_logits: [total_points, num_classes] model predictions
        point_data: dict containing point labels
        
    Returns:
        accuracy: float, accuracy on points
        point_predictions: list of tensors, predictions for each batch's points
        point_labels: list of tensors, labels for each batch's points
    """
    # Get predictions
    predictions = torch.argmax(pred_logits, dim=1)  # [total_points]
    
    # Split predictions and labels by batch
    point_predictions = []
    point_labels = []
    total_correct = 0
    total_points = 0
    
    offset = point_data['offset']
    labels = point_data['labels']
    
    start_idx = 0
    for b in range(len(offset)):
        end_idx = offset[b].item()
        
        # Get predictions and labels for this batch
        batch_point_preds = predictions[start_idx:end_idx]  # [num_points_in_batch]
        batch_point_labels = labels[start_idx:end_idx]  # [num_points_in_batch]
        
        point_predictions.append(batch_point_preds)
        point_labels.append(batch_point_labels)
        
        # Calculate accuracy for this batch
        correct = (batch_point_preds == batch_point_labels).sum().item()
        total_correct += correct
        total_points += batch_point_labels.size(0)
        
        start_idx = end_idx
    
    accuracy = total_correct / total_points if total_points > 0 else 0.0
    
    return accuracy, point_predictions, point_labels


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_point_correct = 0
    total_point_samples = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, (points, masks) in enumerate(pbar):
        points = points.to(device).float()
        masks = masks.to(device).long()

        # Prepare input for Point Transformer
        point_data = prepare_point_transformer_input(points, masks, device)

        optimizer.zero_grad()

        # Forward pass
        data_dict = {
            'coord': point_data['coord'],
            'feat': point_data['feat'],
            'offset': point_data['offset']
        }
        logits = model(data_dict)  # [total_points, num_classes]

        # Calculate loss
        loss = criterion(logits, point_data['labels'])

        # Backward pass
        loss.backward()
        optimizer.step()

        # Calculate point-level accuracy
        point_accuracy, point_predictions, point_labels = calculate_point_accuracy(logits, point_data)

        total_loss += loss.item()
        
        # Calculate point-level metrics
        for batch_point_preds, batch_point_labels in zip(point_predictions, point_labels):
            total_point_correct += (batch_point_preds == batch_point_labels).sum().item()
            total_point_samples += batch_point_labels.size(0)

        # Update progress bar
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
    # Point-level visualization
    point_preds = point_predictions.cpu().numpy()  # [num_points]
    point_labels_np = point_labels.cpu().numpy()  # [num_points]
    original_points_np = original_points.cpu().numpy() if torch.is_tensor(original_points) else original_points  # [num_points, 3]

    # Save PLY files
    sample_name = f"epoch_{epoch:03d}_sample_{sample_idx:02d}"

    # === POINT-LEVEL VISUALIZATIONS ===
    # 1. Save all original points (gray)
    if len(original_points_np) > 0:
        point_input_colors = np.full((len(original_points_np), 3), [128, 128, 128], dtype=np.uint8)
        save_ply(vis_dir / f"{sample_name}_points_input.ply", original_points_np, point_input_colors)
        
        # 2. Save predicted foreground points (green)
        point_pred_fg_indices = np.where(point_preds == 1)[0]
        if len(point_pred_fg_indices) > 0:
            pred_fg_points = original_points_np[point_pred_fg_indices]
            pred_fg_point_colors = np.full((len(point_pred_fg_indices), 3), [0, 255, 0], dtype=np.uint8)
            save_ply(vis_dir / f"{sample_name}_points_pred_fg.ply", pred_fg_points, pred_fg_point_colors)
        else:
            save_ply(vis_dir / f"{sample_name}_points_pred_fg.ply", np.empty((0, 3)))
        
        # 3. Save ground truth foreground points (red)
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


def validate_epoch(model, dataloader, criterion, device, save_vis=False, vis_dir=None, epoch=None):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0.0
    total_point_correct = 0
    total_point_samples = 0

    # Store data for visualization
    vis_data = {'point_predictions': [], 'point_labels': [], 'point_data': []}
    max_vis_batches = 2  # Only save first few batches for visualization

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for batch_idx, (points, masks) in enumerate(pbar):
            points = points.to(device).float()
            masks = masks.to(device).long()

            # Prepare input for Point Transformer
            point_data = prepare_point_transformer_input(points, masks, device)

            # Forward pass
            data_dict = {
                'coord': point_data['coord'],
                'feat': point_data['feat'],
                'offset': point_data['offset']
            }
            logits = model(data_dict)

            # Calculate loss and point-level accuracy
            loss = criterion(logits, point_data['labels'])
            point_accuracy, point_predictions, point_labels = calculate_point_accuracy(logits, point_data)

            total_loss += loss.item()
            
            # Calculate point-level metrics
            for batch_point_preds, batch_point_labels in zip(point_predictions, point_labels):
                total_point_correct += (batch_point_preds == batch_point_labels).sum().item()
                total_point_samples += batch_point_labels.size(0)

            # Collect data for visualization (only first few batches)
            if save_vis and batch_idx < max_vis_batches:
                vis_data['point_predictions'].extend(point_predictions)
                vis_data['point_labels'].extend(point_labels)
                vis_data['point_data'].append(point_data)

            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'PointAcc': f'{point_accuracy:.4f}'
            })

    # Save visualization if requested
    if save_vis and vis_dir is not None and epoch is not None:
        sample_idx = 0
        for batch_idx, point_data_batch in enumerate(vis_data['point_data']):
            # Process each sample in the batch
            batch_size = len(point_data_batch['offset'])
            offset = point_data_batch['offset']
            
            start_idx = 0
            for b in range(min(batch_size, 10)):  # max 10 samples per batch
                if sample_idx >= 10:  # total max samples
                    break
                    
                # Get point range for this sample
                if b < len(offset):
                    end_idx = offset[b].item()
                else:
                    end_idx = point_data_batch['coord'].shape[0]
                
                # Get sample-specific data
                sample_point_preds = vis_data['point_predictions'][sample_idx]
                sample_point_labels = vis_data['point_labels'][sample_idx]
                
                # Get original points for this sample
                original_points_sample = point_data_batch['original_points'][b]  # [num_points, 3]
                
                save_enhanced_validation_visualization(
                    sample_point_preds, sample_point_labels, 
                    original_points_sample, vis_dir, epoch, sample_idx
                )
                
                sample_idx += 1
                start_idx = end_idx

    avg_loss = total_loss / len(dataloader)
    avg_point_accuracy = total_point_correct / total_point_samples

    return avg_loss, avg_point_accuracy


def setup_logging(log_dir):
    """Setup logging configuration"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_dir / 'train.log')
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def main():
    args = get_args()

    # Set random seed
    seed_everything(args.seed)

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup logging
    logger = setup_logging(args.log_dir)
    logger.info(f"Arguments: {vars(args)}")
    logger.info(f"Using device: {device}")
    logger.info(
        "Fixed settings: in_channels=3, num_classes=2, val_every=1"
    )

    # Create datasets
    logger.info("Creating datasets...")
    dataset_cls = ShapeNetDataset if args.dataset_mode == 'single' else ShapeNetMultiClassDataset
    logger.info(f"ShapeNet dataset mode: {args.dataset_mode} ({dataset_cls.__name__})")
    train_dataset = dataset_cls(
        mode='train',
        data_root=args.data_root,
        split_txt_dir=args.split_txt_dir,
        num_point=args.num_point,
        # horizontal_plane_prob = 0
    )

    val_dataset = dataset_cls(
        mode='test',
        data_root=args.data_root,
        split_txt_dir=args.split_txt_dir,
        num_point=args.num_point,
        # horizontal_plane_prob = 0
    )

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")

    # Create data loaders
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

    # Create model
    logger.info("Creating Point Transformer V2 model...")
    model = PointTransformerV2(
        in_channels=3,
        num_classes=2,
        # All other parameters use default values
    ).to(device)
    logger.info(f"Model created successfully \n {model}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Create optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Create directories for saving models and visualizations
    checkpoint_dir = Path(args.log_dir) / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    shapenet_vis_dir = Path(args.log_dir) / 'shapenet_val_vis'
    shapenet_vis_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    logger.info("Starting training...")
    best_val_point_acc = 0.0
    
    # Lists to store metrics for plotting
    shapenet_train_point_accs = []
    shapenet_val_point_accs = []
    shapenet_train_losses = []
    shapenet_val_losses = []
    
    # Track which epochs had validation
    val_epochs = []

    for epoch in range(1, args.epochs + 1):
        logger.info(f"Epoch {epoch}/{args.epochs}")

        # Train
        train_loss, train_point_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Store training metrics
        shapenet_train_point_accs.append(train_point_acc)
        shapenet_train_losses.append(train_loss)

        logger.info(f"Train Loss: {train_loss:.4f}, Train PointAcc: {train_point_acc:.4f}")

        # Validate
        if epoch % 1 == 0:
            # Enable visualization for validation
            val_loss, val_point_acc = validate_epoch(
                model, val_loader, criterion, device,
                save_vis=False, vis_dir=shapenet_vis_dir, epoch=epoch
            )

            # Store validation metrics
            shapenet_val_point_accs.append(val_point_acc)
            shapenet_val_losses.append(val_loss)
            val_epochs.append(epoch)
            
            logger.info(f"ShapeNet Val Loss: {val_loss:.4f}, Val PointAcc: {val_point_acc:.4f}")
            logger.info(f"ShapeNet validation visualization files saved to {shapenet_vis_dir}")

            # Save best model based on point accuracy
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

        # Save checkpoint
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
    
    # Plot training curves
    plot_training_curves(shapenet_train_point_accs, shapenet_val_point_accs,
                        shapenet_train_losses, shapenet_val_losses,
                        val_epochs, args.log_dir)


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
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Point-level accuracy curves
    ax1.plot(epochs, shapenet_train_point_accs, 'b-', label='ShapeNet Train', linewidth=2, marker='o')
    ax1.plot(val_epochs, shapenet_val_point_accs, 'r-', label='ShapeNet Val', linewidth=2, marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Point-Level Accuracy')
    ax1.set_title('Point-Level Accuracy Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Plot 2: Loss curves
    ax2.plot(epochs, shapenet_train_losses, 'b-', label='ShapeNet Train', linewidth=2, marker='o')
    ax2.plot(val_epochs, shapenet_val_losses, 'r-', label='ShapeNet Val', linewidth=2, marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training Loss Curves')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    plot_path = Path(log_dir) / 'training_curves.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to: {plot_path}")


if __name__ == '__main__':
    main()
