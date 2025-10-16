import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

def project_lidar_to_image(lidar_points, intr, extr, img_shape):
    """Проецирует 3D точки на изображение и возвращает (coords_2D, depth)."""
    H, W = img_shape[:2]
    
    # Преобразуем точки в однородные координаты
    points_h = np.hstack((lidar_points, np.ones((lidar_points.shape[0], 1))))
    
    extr = np.linalg.inv(extr)
    
    # Преобразуем точки в систему координат камеры используя extrinsics
    # extr уже преобразует из лидара в камеру, НЕ применяем дополнительную коррекцию осей
    points_cam = (extr @ points_h.T).T  # (N, 4)
    
    # Отфильтровываем точки позади камеры
    mask_z = points_cam[:, 2] > 0
    points_cam_valid = points_cam[mask_z]
        
    if len(points_cam_valid) == 0:
        print("Нет точек перед камерой!")
        return np.array([]), np.array([])
    
    # Проецируем на изображение с помощью intrinsics
    points_2d_h = (intr @ points_cam_valid[:, :3].T).T  # (N, 3)
    
    # Нормализуем однородные координаты
    z = points_2d_h[:, 2]
    x = points_2d_h[:, 0] / z
    y = points_2d_h[:, 1] / z
    
    coords_2d = np.stack([x, y], axis=1)
    
    # Фильтруем точки внутри изображения
    mask_img = (
        (x >= 0) & (x < W) &
        (y >= 0) & (y < H)
    )
    
    coords_final = coords_2d[mask_img]
    depth_final = z[mask_img]
    
    return coords_final, depth_final

def make_vis(imgs, pcs, intrinsic_matrix, gt_T_to_camera, T_init, T_pred, vis_dir, save_dir):
    # === Берём только первый сэмпл из батча ===
    img = np.array(imgs[0])
    lidar_points = pcs[0].detach().cpu().numpy()
    intr_gt = intrinsic_matrix[0].detach().cpu().numpy()
    extr_gt = gt_T_to_camera[0].detach().cpu().numpy()
    extr_init = T_init[0].detach().cpu().numpy()
    extr_pred = T_pred[0].detach().cpu().numpy()
    
    # === GT проекция ===
    coords_gt, depth_gt = project_lidar_to_image(lidar_points, intr_gt, extr_gt, img.shape)
    img_gt = img.copy()
    
    for (u, v), z in zip(coords_gt.astype(int), depth_gt):
        # Цвет в зависимости от глубины
        intensity = min(z / 50.0, 1.0)
        color = (int(255 * (1 - intensity)), int(255 * intensity), 255)
        cv2.circle(img_gt, (int(u), int(v)), 1, color, -1)
        
    # === Init проекция ===
    coords_init, depth_init = project_lidar_to_image(lidar_points, intr_gt, extr_init, img.shape)
    img_init = img.copy()
    
    for (u, v), z in zip(coords_init.astype(int), depth_init):
        # Цвет в зависимости от глубины
        intensity = min(z / 50.0, 1.0)
        color = (int(255 * (1 - intensity)), int(255 * intensity), 255)
        cv2.circle(img_init, (int(u), int(v)), 1, color, -1)
        
    # === BEVCalib проекция ===
    coords_pred, depth_pred = project_lidar_to_image(lidar_points, intr_gt, extr_pred, img.shape)
    img_pred = img.copy()
    
    for (u, v), z in zip(coords_pred.astype(int), depth_pred):
        # Цвет в зависимости от глубины
        intensity = min(z / 50.0, 1.0)
        color = (255, int(255 * intensity), int(255 * (1 - intensity)))
        cv2.circle(img_pred, (int(u), int(v)), 1, color, -1)

    # Визуализация
    plt.figure(figsize=(12, 12))

    plt.subplot(3, 1, 1)
    plt.imshow(cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(f"GT KITTI (точек: {len(coords_gt)})", fontsize=14)
    
    plt.subplot(3, 1, 2)
    plt.imshow(cv2.cvtColor(img_init, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(f"Initial (точек: {len(coords_init)})", fontsize=14)

    plt.subplot(3, 1, 3)
    plt.imshow(cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(f"BEVCalib Predicted (точек: {len(coords_pred)})", fontsize=14)

    plt.tight_layout()
    panel_path = os.path.join(vis_dir, save_dir)
    os.makedirs(vis_dir, exist_ok=True)
    plt.savefig(panel_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"Визуализация сохранена: {panel_path}")
