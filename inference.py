import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from generic_dataset import GenericDataset, collate_fn
from bevcalib.bev_calib import BEVCalib
from torch.utils.tensorboard import SummaryWriter
import argparse
from datetime import datetime
import numpy as np
from utils.tools import generate_single_perturbation_from_T
from metrics.metrics import calc_metrics
import cv2
import os


def parse_args():
    parser = argparse.ArgumentParser("Run inference / evaluation with generic dataset")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to directory containing images")
    parser.add_argument("--pointcloud_dir", type=str, required=True,
                        help="Path to directory containing pointcloud files")
    parser.add_argument("--calib_data", type=str, default=None, help="Path to calibration file or directory")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the saved .pth checkpoint")
    parser.add_argument("--log_dir", type=str, default="./logs/inference")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--xyz_only", type=int, default=1)
    parser.add_argument("--angle_range_deg", type=float, default=20.0)
    parser.add_argument("--trans_range", type=float, default=1.5)
    return parser.parse_args()


def load_calibration_data(calib_path):
    """Load calibration data from file or directory"""
    if calib_path is None:
        return None

    if os.path.isfile(calib_path):
        # Single calibration file
        try:
            calibration = np.load(calib_path)
            return {'default': (calibration['K'], calibration['T'])}
        except:
            print(f"Warning: Could not load calibration from {calib_path}")
            return None
    elif os.path.isdir(calib_path):
        # Directory with multiple calibration files
        calib_data = {}
        for calib_file in os.listdir(calib_path):
            if calib_file.endswith('.npz'):
                file_path = os.path.join(calib_path, calib_file)
                try:
                    calibration = np.load(file_path)
                    key = calib_file.replace('.npz', '')
                    calib_data[key] = (calibration['K'], calibration['T'])
                except:
                    print(f"Warning: Could not load calibration from {file_path}")
        return calib_data if calib_data else None
    return None


def main():
    args = parse_args()
    xyz_only_choice = args.xyz_only > 0

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(args.log_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # Load calibration data
    calib_data = load_calibration_data(args.calib_data)

    # Create dataset
    dataset = GenericDataset(
        image_dir=args.image_dir,
        pointcloud_dir=args.pointcloud_dir,
        calib_data=calib_data
    )

    loader = DataLoader(
        dataset,
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

    eval_angle = np.array([args.angle_range_deg])
    eval_trans_range = np.array([args.trans_range])

    for angle, trans in zip(eval_angle, eval_trans_range):
        print(f"\nEvaluating perturb   angle {angle},  trans {trans}")
        step = 0

        with torch.no_grad():
            for b_idx, (imgs, pcs, masks, gt_T_to_camera, intrinsics) in enumerate(loader):
                gt_T_to_camera = np.array(gt_T_to_camera).astype(np.float32)
                init_T_to_camera, ang_err, trans_err = generate_single_perturbation_from_T(
                    gt_T_to_camera, angle_range_deg=eval_angle, trans_range=eval_trans_range
                )

                resize_imgs = torch.from_numpy(np.array(imgs)).permute(0, 3, 1, 2).float().to(device)

                if xyz_only_choice:
                    pcs = np.array(pcs)[:, :, :3]
                pcs = torch.from_numpy(np.array(pcs)).float().to(device)

                gt_T_to_camera = torch.from_numpy(gt_T_to_camera).float().to(device)
                init_T_to_camera = torch.from_numpy(init_T_to_camera).float().to(device)
                post_cam2ego_T = torch.eye(4).unsqueeze(0).repeat(gt_T_to_camera.shape[0], 1, 1).float().to(device)
                intrinsic_matrix = torch.from_numpy(np.array(intrinsics)).float().to(device)

                T_pred, init_loss, loss = model(
                    resize_imgs, pcs, gt_T_to_camera, init_T_to_camera,
                    post_cam2ego_T, intrinsic_matrix, masks=masks, out_init_loss=False
                )

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
