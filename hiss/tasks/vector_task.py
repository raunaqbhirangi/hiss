import numpy as np
import torch

from hiss.tasks.task import Task


class VectorTask(Task):
    def __init__(self, filter_input=False, filter_cutoff=0.03) -> None:
        super().__init__("VectorTask")
        self.filter_input = filter_input
        self.metrics = [
            "mse",
            "err",
            "err_x",
            "err_y",
            "cum_err",
        ]
        self.filter_cutoff = filter_cutoff
        self.input_eid_key = "gyro_episode_ids"
        self.target_eid_key = "pose_episode_ids"
        self.predict_diffs = True

    def input_tf_fn(self, obs_dict):
        acc_data = obs_dict["acce"]
        gyro_data = obs_dict["gyro"]
        return np.concatenate((acc_data, gyro_data), axis=-1)

    def target_tf_fn(self, obs_dict):
        pos = obs_dict["pose"][:, :3]
        quat = obs_dict["pose"][:, 3:]
        return pos

    def compute_input_shift_scale(self, input_list):
        # Don't normalize rotation matrix columns
        input_mean = 0.0
        input_min = torch.amin(torch.cat(input_list, dim=0), dim=0)
        input_max = torch.amax(torch.cat(input_list, dim=0), dim=0)
        input_std = (input_max - input_min) / 5
        return input_mean, input_std

    def compute_target_shift_scale(self, target_list):
        target_mean, target_std = super().compute_target_shift_scale(target_list)
        return target_mean, target_std  # * 3

    def compute_metrics(self, preds, targets, out_std):
        err = (preds - targets) * out_std
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
