import numpy as np
from hiss.tasks.task import Task


class MarkerWritingTask(Task):
    def __init__(self, filter_input=False, filter_cutoff=0.03) -> None:
        super().__init__("MarkerWritingTask")
        self.metrics = ["mse", "err", "err_x", "err_y", "cum_err"]
        self.filter_input = filter_input
        self.filter_cutoff = filter_cutoff
        self.input_eid_key = "tactile_episode_ids"
        self.target_eid_key = "kinova_episode_ids"
        self.predict_diffs = True

    def input_tf_fn(self, obs_dict):
        tactile_data = obs_dict["tactile"]
        return tactile_data

    def target_tf_fn(self, obs_dict):
        kinova_pos = obs_dict["kinova"][:, :3]
        kinova_quat = obs_dict["kinova"][:, 3:]
        return kinova_pos[:, :2]

    def compute_input_shift_scale(self, input_list):
        # Don't normalize rotation matrix columns
        input_mean, input_std = super().compute_input_shift_scale(input_list)
        input_std *= 3
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
