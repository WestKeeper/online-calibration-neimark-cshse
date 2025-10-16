import torch
from utils.tools import rotation_matrix_to_euler_xyz


def calc_metrics(T_pred, gt_T_to_camera):
    translation_error = torch.abs((T_pred[:, :3, 3] - gt_T_to_camera[:, :3, 3]).reshape(T_pred.shape[0], 3))
    rotation_error = torch.abs(
        torch.stack(rotation_matrix_to_euler_xyz(T_pred[:, :3, :3] @ gt_T_to_camera[:, :3, :3].transpose(-2, -1)),
                    dim=0).reshape(T_pred.shape[0], 3))

    return translation_error, rotation_error
