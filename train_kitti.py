import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.kitti_dataset import KittiDataset, collate_fn
from bevcalib.bev_calib import BEVCalib
from metrics.metrics import calc_metrics
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import argparse
from datetime import datetime
from torch.utils.data import random_split, Subset
import numpy as np
from pathlib import Path
from utils.tools import generate_single_perturbation_from_T
import shutil
import cv2
import os
import sys
import logging

def setup_logging(log_dir):
    log_file = os.path.join(log_dir, 'training.log')
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    class LoggerWriter:
        def __init__(self, level):
            self.level = level
            
        def write(self, message):
            if message.strip():
                self.level(message)
                
        def flush(self):
            pass
    
    sys.stdout = LoggerWriter(logging.info)
    sys.stderr = LoggerWriter(logging.warning)

def parse_args():
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

    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument("--dataset_root", type=str, default="./datasets/kitti-odemetry")
    parser.add_argument("--log_dir", type=str, default="./logs/train")
    parser.add_argument("--save_ckpt_per_epoches", type=int, default=-1)
    parser.add_argument("--label", type=str, default=None)

    parser.add_argument("--angle_range_deg", type=float,
                        default=config_dict.get('angle_range_deg', None))
    parser.add_argument("--trans_range", type=float,
                        default=config_dict.get('trans_range', None))
    parser.add_argument("--eval_angle_range_deg", type=float,
                        default=config_dict.get('eval_angle_range_deg', None))
    parser.add_argument("--eval_trans_range", type=float,
                        default=config_dict.get('eval_trans_range', None))
    parser.add_argument("--num_epochs", type=int,
                        default=config_dict.get('num_epochs', 1))
    parser.add_argument("--eval_epoches", type=int,
                        default=config_dict.get('eval_epoches', 10))
    parser.add_argument("--deformable", type=int,
                        default=config_dict.get('deformable', -1))
    parser.add_argument("--bev_encoder", type=int,
                        default=config_dict.get('bev_encoder', 1))
    parser.add_argument("--xyz_only", type=int,
                        default=config_dict.get('xyz_only', 1))
    parser.add_argument("--batch_size", type=int,
                        default=config_dict.get('batch_size', 4))
    parser.add_argument("--lr", type=float,
                        default=config_dict.get('lr', 1e-4))
    parser.add_argument("--wd", type=float,
                        default=config_dict.get('wd', 1e-4))
    parser.add_argument("--step_size", type=int,
                        default=config_dict.get('step_size', 80))
    parser.add_argument("--scheduler", type=int,
                        default=config_dict.get('scheduler', 1))
    parser.add_argument("--train_val_split", type=float,
                        default=config_dict.get('train_val_split', 0.7))
    parser.add_argument("--train_test_split", type=float,
                        default=config_dict.get('train_test_split', 0.8))

    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--perturb_seed", type=int, default=-1)
    parser.add_argument("--num_samples", type=int, default=0)
    parser.add_argument("--pretrain_ckpt", type=str, default=None)
    parser.add_argument("--config", type=str, default=config_args.config)

    return parser.parse_args()

def main():
    args = parse_args()
    
    log_dir = args.log_dir
    if args.label is not None:
        log_dir = os.path.join(log_dir, args.label)
    else:
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = f"{log_dir}/{current_time}"
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    setup_logging(log_dir)
    
    print(args)
    
    num_epochs = args.num_epochs
    dataset_root = args.dataset_root
    
    model_save_dir = os.path.join(log_dir, "model")
    ckpt_save_dir = os.path.join(log_dir, "checkpoint")
    if not os.path.exists(ckpt_save_dir) or args.save_ckpt_per_epoches > 0:
        os.makedirs(ckpt_save_dir)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    bev_calib_dir = os.path.join(current_dir, 'bevcalib')
    shutil.copytree(bev_calib_dir, os.path.join(log_dir, 'bevcalib'))
    
    writer = SummaryWriter(log_dir)
    dataset = KittiDataset(dataset_root, ['00', '01', '02', '03'])

    generator = torch.Generator().manual_seed(args.split_seed)

    if args.num_samples > 0 and args.num_samples < len(dataset):
        indices = torch.randperm(len(dataset), generator=generator)
        subset_indices = indices[:args.num_samples]
        dataset = Subset(dataset, subset_indices)

    total_size = len(dataset)
    train_val_size = int(args.train_test_split * total_size)
    test_size = total_size - train_val_size

    train_size = int(args.train_val_split * train_val_size)
    val_size = train_val_size - train_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        collate_fn=collate_fn,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        collate_fn=collate_fn,
    )

    deformable_choise = args.deformable > 0
    bev_encoder_choise = args.bev_encoder > 0
    xyz_only_choise = args.xyz_only > 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BEVCalib(
        deformable=deformable_choise,
        bev_encoder=bev_encoder_choise
    ).to(device)

    if args.pretrain_ckpt is not None:
        state_dict = torch.load(args.pretrain_ckpt, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'], strict=True)
        print(f"Load pretrain model from {args.pretrain_ckpt}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    scheduler_choice = args.scheduler > 0
    if scheduler_choice:
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=0.5)

    train_noise = {
        "angle_range_deg": args.angle_range_deg if args.angle_range_deg is not None else 20,
        "trans_range": args.trans_range if args.trans_range is not None else 1.5,
    }

    eval_noise = {
        "angle_range_deg": args.eval_angle_range_deg if args.eval_angle_range_deg is not None else train_noise["angle_range_deg"],
        "trans_range": args.eval_trans_range if args.eval_trans_range is not None else train_noise["trans_range"],
    }
    
    if args.perturb_seed > 0:
        np.random.seed(args.perturb_seed)

    for epoch in range(num_epochs):
        model.train()
        train_loss = {}
        out_init_loss_choice = False
        if epoch < 5:
            out_init_loss_choice = True
        for batch_index, (imgs, pcs, masks, gt_T_to_camera, intrinsics) in enumerate(train_loader):
            gt_T_to_camera = np.array(gt_T_to_camera).astype(np.float32)
            init_T_to_camera, _, _ = generate_single_perturbation_from_T(gt_T_to_camera, angle_range_deg=train_noise["angle_range_deg"], trans_range=train_noise["trans_range"])
            resize_imgs = torch.from_numpy(np.array(imgs)).permute(0, 3, 1, 2).float().to(device)
            if xyz_only_choise:
                pcs = np.array(pcs)[:, :, :3]
            pcs = torch.from_numpy(np.array(pcs)).float().to(device)
            gt_T_to_camera = torch.from_numpy(gt_T_to_camera).float().to(device)
            init_T_to_camera = torch.from_numpy(init_T_to_camera).float().to(device)
            post_cam2ego_T = torch.eye(4).unsqueeze(0).repeat(gt_T_to_camera.shape[0], 1, 1).float().to(device)
            intrinsic_matrix = torch.from_numpy(np.array(intrinsics)).float().to(device)

            optimizer.zero_grad()
            T_pred, init_loss, loss = model(resize_imgs, pcs, gt_T_to_camera, init_T_to_camera, post_cam2ego_T, intrinsic_matrix, masks=masks, out_init_loss=out_init_loss_choice)
            total_loss = loss["total_loss"]
            total_loss.backward()
            optimizer.step()
            for key in loss.keys():
                if key not in train_loss.keys():
                    train_loss[key] = loss[key].item()
                else:
                    train_loss[key] += loss[key].item()
            
            if init_loss is not None:
                for key in init_loss.keys():
                    train_key = f"init_{key}"
                    if train_key not in train_loss.keys():
                        train_loss[train_key] = init_loss[key].item()
                    else:
                        train_loss[train_key] += init_loss[key].item()

            if batch_index % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_index+1}/{len(train_loader)}], Loss: {total_loss.item():.4f}")

        if scheduler_choice:   
            scheduler.step()    
        
        for key in train_loss.keys():
            train_loss[key] /= len(train_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss {key}: {train_loss[key]:.4f}")
            writer.add_scalar(f"Loss/train/{key}", train_loss[key], epoch)
        
        if epoch == num_epochs - 1 or (args.save_ckpt_per_epoches > 0 and (epoch + 1) % args.save_ckpt_per_epoches == 0):
            ckpt_path = os.path.join(ckpt_save_dir, f"ckpt_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_noise': train_noise,
                'eval_noise': eval_noise,
                'args': vars(args) 
            }, ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")

        train_loss = None
        init_loss = None
        loss = None

        translation_errors, rotation_errors = [], []

        if (epoch + 1) % args.eval_epoches == 0:
            eval_trans_range = eval_noise["trans_range"]
            eval_angle_range = eval_noise["angle_range_deg"]
            model.eval()
            val_loss = {}
            with torch.no_grad():
                for batch_index, (imgs, pcs, masks, gt_T_to_camera, intrinsics) in enumerate(val_loader):
                    gt_T_to_camera = np.array(gt_T_to_camera).astype(np.float32)
                    init_T_to_camera, ang_err, trans_err = generate_single_perturbation_from_T(gt_T_to_camera, angle_range_deg=eval_angle_range, trans_range=eval_trans_range)
                    resize_imgs = torch.from_numpy(np.array(imgs)).permute(0, 3, 1, 2).float().to(device)
                    if xyz_only_choise:
                        pcs = np.array(pcs)[:, :, :3]
                    pcs = torch.from_numpy(np.array(pcs)).float().to(device)
                    gt_T_to_camera = torch.from_numpy(gt_T_to_camera).float().to(device)
                    init_T_to_camera = torch.from_numpy(init_T_to_camera).float().to(device)
                    post_cam2ego_T = torch.eye(4).unsqueeze(0).repeat(gt_T_to_camera.shape[0], 1, 1).float().to(device)
                    intrinsic_matrix = torch.from_numpy(np.array(intrinsics)).float().to(device)
                    T_pred, init_loss, loss = model(resize_imgs, pcs, gt_T_to_camera, init_T_to_camera, post_cam2ego_T, intrinsic_matrix, masks=masks, out_init_loss=False)

                    for key in loss.keys():
                        val_key = key
                        if val_key not in val_loss.keys():
                            val_loss[val_key] = loss[key].item()
                        else:
                            val_loss[val_key] += loss[key].item()
                    if init_loss is not None:
                        for key in init_loss.keys():
                            val_key = f"init_{key}"
                            if val_key not in val_loss.keys():
                                val_loss[val_key] = init_loss[key].item()
                            else:
                                val_loss[val_key] += init_loss[key].item()

                    translation_error, rotation_error = calc_metrics(T_pred, gt_T_to_camera)
                    translation_errors.append(translation_error)
                    rotation_errors.append(rotation_error)

            print(f"Epoch [{epoch + 1}/{num_epochs}], {eval_angle_range}_{eval_trans_range}")

            for key in val_loss.keys():
                val_loss[key] /= len(val_loader)
                print(f"  Validation Loss {key}: {val_loss[key]:.4f}")
                writer.add_scalar(f"Loss/val/{key}", val_loss[key], epoch)

            translation_errors = torch.cat(translation_errors, dim=0).cpu().numpy()
            rotation_errors = torch.cat(rotation_errors, dim=0).cpu().numpy()

            print("  Validation translation xyz error: ", np.mean(translation_errors, axis=0))
            print("  Validation rotation ypr error: ", np.mean(rotation_errors, axis=0))
            
            writer.add_scalar(f"Metric/val/error/x", np.mean(translation_errors, axis=0)[0], epoch)
            writer.add_scalar(f"Metric/val/error/y", np.mean(translation_errors, axis=0)[1], epoch)
            writer.add_scalar(f"Metric/val/error/z", np.mean(translation_errors, axis=0)[2], epoch)
            
            writer.add_scalar(f"Metric/val/error/yaw", np.mean(rotation_errors, axis=0)[0], epoch)
            writer.add_scalar(f"Metric/val/error/pitch", np.mean(rotation_errors, axis=0)[1], epoch)
            writer.add_scalar(f"Metric/val/error/roll", np.mean(rotation_errors, axis=0)[2], epoch)

            val_loss = None
            loss = None

    writer.close()
    print(f"Logs are saved at {log_dir}")

    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

if __name__ == "__main__":
    main()
