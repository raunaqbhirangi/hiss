import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from hiss.tasks.task import Task


class TotalCaptureTask(Task):
    def __init__(self, filter_input=False, filter_cutoff=0.03) -> None:
        super().__init__("TotalCaptureTask")
        self.filter_input = filter_input
        self.metrics = [
            "mse",
            "err",
            "err_x",
            "err_y",
            "cum_err",
        ]
        self.input_list = [
            "Head",
            "Sternum",
            "Pelvis",
            "L_UpArm",
            "R_UpArm",
            "L_LowArm",
            "R_LowArm",
            "L_UpLeg",
            "R_UpLeg",
            "L_LowLeg",
            "R_LowLeg",
            "L_Foot",
            "R_Foot",
        ]
        self.target_list = [
            "Hips",
            "Spine",
            "Spine1",
            "Spine2",
            "Spine3",
            "Neck",
            "Head",
            "RightShoulder",
            "RightArm",
            "RightForeArm",
            "RightHand",
            "LeftShoulder",
            "LeftArm",
            "LeftForeArm",
            "LeftHand",
            "RightUpLeg",
            "RightLeg",
            "RightFoot",
            "LeftUpLeg",
            "LeftLeg",
            "LeftFoot",
        ]
        self.filter_cutoff = filter_cutoff
        self.input_eid_key = "rot_global_frame_Head_episode_ids"
        self.target_eid_key = "pose_Head_episode_ids"
        self.predict_diffs = True

    def input_tf_fn(self, obs_dict):
        acc_data = np.concatenate(
            [
                obs_dict[f"acce_global_frame_{sensor_name}"]
                for sensor_name in self.input_list
            ],
            axis=-1,
        )
        # Convert gyro data to rotation matrices
        rot_data = []
        for sensor_name in self.input_list:
            gyro_data = obs_dict[f"rot_global_frame_{sensor_name}"]
            rot_matrices = R.from_quat(gyro_data).as_matrix()
            rot_data.append(rot_matrices[:, :, :2].reshape(-1, 6))
        rot_data = np.concatenate(rot_data, axis=-1)
        return np.concatenate((acc_data, rot_data), axis=-1)

    def target_tf_fn(self, obs_dict):
        pose_data = np.concatenate(
            [obs_dict[f"pose_{joint}"][:, :3] for joint in self.target_list], axis=-1
        )
        return pose_data

    def compute_input_shift_scale(self, input_list):
        # Don't normalize rotation matrix columns
        input_mean = 0.0
        input_min = torch.amin(torch.cat(input_list, dim=0), dim=0)
        input_max = torch.amax(torch.cat(input_list, dim=0), dim=0)
        input_std = input_max - input_min
        input_std[
            len(self.input_list) * 3 : len(self.input_list) * 3
            + len(self.input_list) * 9
        ] = 1.0
        return input_mean, input_std

    def compute_target_shift_scale(self, target_list):
        target_mean, target_std = super().compute_target_shift_scale(target_list)
        return target_mean, target_std * 3

    def compute_metrics(self, preds, targets, out_std):
        err = (preds - targets) * out_std * 0.0254  # convert from in to m
        cum_err = np.cumsum(err, axis=0)
        return np.stack(
            (
                np.mean(np.square(err), axis=-1),
                np.linalg.norm(err, axis=-1),
                np.abs(err[:, 0]),
                np.abs(err[:, 1]),
                np.linalg.norm(cum_err, axis=-1),
            ),
            axis=-1,
        )
