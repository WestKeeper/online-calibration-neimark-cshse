import numpy as np
import open3d as o3d
from PIL import Image
from torch.utils.data import Dataset
from utils.tools import crop_and_resize
import os


class GenericDataset(Dataset):
    def __init__(self, image_dir, pointcloud_dir, calib_data=None, suf='.png'):
        """
        Generic dataset for image-pointcloud pairs

        Args:
            image_dir: Path to directory containing images
            pointcloud_dir: Path to directory containing pointcloud files
            calib_data: Either:
                       - Dict with keys as sequence names and values as (K, T) tuples
                       - Path to calibration file
                       - None if no calibration needed
            suf: Image file suffix (default: .png)
        """
        self.all_files = []
        self.image_dir = image_dir
        self.pointcloud_dir = pointcloud_dir
        self.calib_data = calib_data
        self.suf = suf
        self.K = {}
        self.T = {}

        # Load calibration data if provided
        if calib_data is not None:
            if isinstance(calib_data, dict):
                self.K = {k: v[0] for k, v in calib_data.items()}
                self.T = {k: v[1] for k, v in calib_data.items()}
            elif isinstance(calib_data, str) and os.path.exists(calib_data):
                self._load_calibration_from_file(calib_data)

        # Find all valid image-pointcloud pairs
        image_files = os.listdir(image_dir)
        image_files.sort()

        for image_name in image_files:
            if not image_name.endswith(suf):
                continue

            base_name = image_name.replace(suf, '')
            pointcloud_path = os.path.join(pointcloud_dir, base_name + '.bin')

            if not os.path.exists(pointcloud_path):
                continue

            self.all_files.append(base_name)

    def _load_calibration_from_file(self, calib_path):
        """Load calibration from file - implement based on your calibration format"""
        # This should be implemented based on your specific calibration file format
        # For now, we assume it's a numpy .npz file with 'K' and 'T' arrays
        try:
            calibration = np.load(calib_path)
            self.K['default'] = calibration['K']
            self.T['default'] = calibration['T']
        except:
            print(f"Warning: Could not load calibration from {calib_path}")

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        base_name = self.all_files[idx]
        img_path = os.path.join(self.image_dir, base_name + self.suf)
        pcd_path = os.path.join(self.pointcloud_dir, base_name + '.bin')

        if not os.path.exists(img_path) or not os.path.exists(pcd_path):
            print(f'File not exist: {img_path} or {pcd_path}')
            raise FileNotFoundError

        # Load image
        img = Image.open(img_path)
        img = img.resize((1242, 375))  # Standard KITTI size, adjust if needed

        # Load pointcloud
        pcd = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 4)

        # Filter points (adjust thresholds as needed)
        valid_ind = (pcd[:, 0] < -3.) | (pcd[:, 0] > 3.) | (pcd[:, 1] < -3.) | (pcd[:, 1] > 3.)
        pcd = pcd[valid_ind, :]

        # Get calibration data
        seq = 'default'  # Use default if no sequence-specific calibration
        if base_name in self.K:
            seq = base_name
        elif len(self.K) == 0:
            # If no calibration provided, return identity matrices
            gt_transform = np.eye(4)
            intrinsic = np.eye(3)
            return img, pcd, gt_transform, intrinsic

        gt_transform = self.T.get(seq, np.eye(4))
        intrinsic = self.K.get(seq, np.eye(3))

        return img, pcd, gt_transform, intrinsic


def collate_fn(batch):
    target_size = (704, 256)
    processed_data = [crop_and_resize(item[0], target_size, item[3], False) for item in batch]
    imgs = [item[0] for item in processed_data]
    intrinsics = [item[1] for item in processed_data]

    gt_T_to_camera = [item[2] for item in batch]
    pcs = []
    masks = []
    max_num_points = 0

    for item in batch:
        max_num_points = max(max_num_points, item[1].shape[0])

    for item in batch:
        pc = item[1]
        masks.append(np.concatenate([np.ones(pc.shape[0]), np.zeros(max_num_points - pc.shape[0])], axis=0))
        if pc.shape[0] < max_num_points:
            pc = np.concatenate([pc, np.full((max_num_points - pc.shape[0], pc.shape[1]), 999999)], axis=0)
        pcs.append(pc)

    return imgs, pcs, masks, gt_T_to_camera, intrinsics
