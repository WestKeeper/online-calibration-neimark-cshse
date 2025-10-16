import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
from datetime import datetime
from torch.utils.data import random_split
import numpy as np
import cv2
import os
import sys

# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from datasets.kitti_dataset import KittiDataset, collate_fn
from bevcalib.bev_calib import BEVCalib
from metrics.metrics import calc_metrics
from utils.tools import generate_single_perturbation_from_T
from viz import make_vis

def parse_args():
    # Первый парсер для получения пути к конфигу
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_args, _ = config_parser.parse_known_args()

    config_dict = {}
    if config_args.config is not None and os.path.exists(config_args.config):
        with open(config_args.config, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    config_dict[key] = value

    # Основной парсер
    parser = argparse.ArgumentParser(description="Test")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--log_dir", type=str, required=True)

    # Параметры из конфига
    parser.add_argument("--batch_size", type=int,
                        default=config_dict.get('batch_size', 4))
    parser.add_argument("--xyz_only", type=int,
                        default=config_dict.get('xyz_only', 1))
    parser.add_argument("--test_angle_range_deg", type=float,
                        default=config_dict.get('test_angle_range_deg', None))
    parser.add_argument("--test_trans_range", type=float,
                        default=config_dict.get('test_trans_range', None))
    parser.add_argument("--train_val_split", type=float,
                        default=config_dict.get('train_val_split', 0.7))
    parser.add_argument("--train_test_split", type=float,
                        default=config_dict.get('train_test_split', 0.8))

    # Параметры вне конфига
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--perturb_seed", type=int, default=-1)
    parser.add_argument("--num_samples", type=int, default=0)
    parser.add_argument("--vizualize", type=int, default=0)
    parser.add_argument("--config", type=str, default=config_args.config)

    return parser.parse_args()

def main():
    args = parse_args()
    xyz_only_choise = args.xyz_only > 0

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(args.log_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    vis_dir = os.path.join(log_dir, "projections")
    os.makedirs(vis_dir, exist_ok=True)

    dataset = KittiDataset(args.dataset_root, ['00', '01', '02', '03'])
    generator = torch.Generator().manual_seed(args.split_seed)

    # Если num_samples > 0, выбираем случайное подмножество данных
    if args.num_samples > 0 and args.num_samples < len(dataset):
        # Создаем индексы для всего датасета
        indices = torch.randperm(len(dataset), generator=generator)
        # Выбираем только num_samples случайных элементов
        subset_indices = indices[:args.num_samples]
        # Создаем подмножество датасета
        from torch.utils.data import Subset
        dataset = Subset(dataset, subset_indices)

    total_size = len(dataset)
    train_val_size = int(args.train_test_split * total_size)
    test_size = total_size - train_val_size

    # Внутри train_val части вычисляем размеры для train и val
    train_size = int(args.train_val_split * train_val_size)
    val_size = train_val_size - train_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    print(f"Test dataset size: {len(test_dataset)}")

    # Используем только test_dataset
    loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BEVCalib(
        deformable=False,      
        bev_encoder=True,
    ).to(device)

    ckpt = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    total_losses = []
    translation_losses = []
    rotation_losses = []
    quant_losses = []
    reproj_losses = []

    translation_errors = []
    rotation_errors = []

    eval_angle = np.array([args.test_angle_range_deg])
    eval_trans_range = np.array([args.test_trans_range])
    
    if args.perturb_seed > 0:
        np.random.seed(args.perturb_seed)

    for angle, trans in zip(eval_angle, eval_trans_range):
        print(f"\nEvaluating perturb   angle {angle},  trans {trans}")
        step = 0

        with torch.no_grad():
            for b_idx, (imgs, pcs, masks, gt_T_to_camera, intrinsics) in enumerate(loader):
                gt_T_to_camera = np.array(gt_T_to_camera).astype(np.float32)
                init_T_to_camera, ang_err, trans_err = generate_single_perturbation_from_T(gt_T_to_camera, angle_range_deg=eval_angle, trans_range=eval_trans_range)
                resize_imgs = torch.from_numpy(np.array(imgs)).permute(0, 3, 1, 2).float().to(device)
                if xyz_only_choise:
                    pcs = np.array(pcs)[:, :, :3]
                pcs = torch.from_numpy(np.array(pcs)).float().to(device)
                gt_T_to_camera = torch.from_numpy(gt_T_to_camera).float().to(device)
                init_T_to_camera = torch.from_numpy(init_T_to_camera).float().to(device)
                post_cam2ego_T = torch.eye(4).unsqueeze(0).repeat(gt_T_to_camera.shape[0], 1, 1).float().to(device)
                intrinsic_matrix = torch.from_numpy(np.array(intrinsics)).float().to(device)
                T_pred, init_loss, loss = model(resize_imgs, pcs, gt_T_to_camera, init_T_to_camera, post_cam2ego_T, intrinsic_matrix, masks=masks, out_init_loss=False)
                
                # ===== VIS ======
                if args.vizualize > 0:
                    make_vis(imgs, pcs, intrinsic_matrix, gt_T_to_camera, init_T_to_camera, T_pred, vis_dir, f"{b_idx:05d}_vis.png")
                # ================
                
                metrics = {k: v.item() for k, v in loss.items()}
                if init_loss:
                    metrics.update({f"init_{k}": v.item() for k, v in init_loss.items()})

                total_losses.append(metrics["total_loss"])
                translation_losses.append(metrics["translation_loss"])
                rotation_losses.append(metrics["rotation_loss"])
                quant_losses.append(metrics["quat_norm_loss"])
                reproj_losses.append(metrics["PC_reproj_loss"])

                translation_error, rotation_error = calc_metrics(T_pred, gt_T_to_camera)

                translation_errors.append(translation_error)
                rotation_errors.append(rotation_error)

                # TensorBoard
                for k, v in metrics.items():
                    writer.add_scalar(f"val/{angle}_{trans}/{k}", v, step)

                print(f"Batch {b_idx:04d} | " +
                      " | ".join([f"{k}: {v:.6f}" for k, v in metrics.items()]))
                step += 1

    writer.close()

    print("\nInference finished. Logs: ", log_dir)
    print("Average losses:")
    print(len(total_losses))
    print(f"Total loss: {np.mean(total_losses):.6f}")
    print(f"Translation loss: {np.mean(translation_losses):.6f}")
    print(f"Rotation loss: {np.mean(rotation_losses):.6f}")
    print(f"Quantization loss: {np.mean(quant_losses):.6f}")
    print(f"Reprojection loss: {np.mean(reproj_losses):.6f}")

    print("STD losses:")
    print(f"Total loss: {np.std(total_losses):.6f}")
    print(f"Translation loss: {np.std(translation_losses):.6f}")
    print(f"Rotation loss: {np.std(rotation_losses):.6f}")
    print(f"Quantization loss: {np.std(quant_losses):.6f}")
    print(f"Reprojection loss: {np.std(reproj_losses):.6f}")

    print("\n")
    print("=" * 50)
    print("Errors")
    print("=" * 50)

    translation_errors = torch.cat(translation_errors, dim=0).cpu().numpy()
    rotation_errors = torch.cat(rotation_errors, dim=0).cpu().numpy()

    print("Average translation xyz error: ", np.mean(translation_errors, axis=0))
    print("Average rotation ypr error: ", np.mean(rotation_errors, axis=0))

    print("STD of translation xyz error: ", np.std(translation_errors, axis=0))
    print("STD of rotation ypr error: ", np.std(rotation_errors, axis=0))


if __name__ == "__main__":
    main()
