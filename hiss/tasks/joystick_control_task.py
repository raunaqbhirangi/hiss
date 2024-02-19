import numpy as np

from hiss.tasks.task import Task


class JoystickControlTask(Task):
    def __init__(self, filter_input=False, filter_cutoff=0.03) -> None:
        super().__init__("JoystickControlTask")
        self.filter_input = filter_input
        self.dset_split = "random"
        self.metrics = [
            "mse",
            "err",
            "err_x",
            "err_y",
            "err_z",
        ]
        self.filter_cutoff = filter_cutoff
        self.input_eid_key = "xela_episode_ids"
        self.target_eid_key = "extreme3d_episode_ids"
        self.predict_diffs = False

    def input_tf_fn(self, obs_dict):
        return np.clip(np.array(obs_dict["xela"]), -1000.0, 1000.0)

    def target_tf_fn(self, obs_dict):
        return np.array(obs_dict["extreme3d"][:, :3])

    def compute_input_shift_scale(self, input_list):
        input_mean = 0.0
        input_std = 1000.0

        return input_mean, input_std

    def compute_target_shift_scale(self, target_list):
        target_mean, target_std = super().compute_target_shift_scale(target_list)
        return target_mean, target_std * 2

    def compute_metrics(self, preds, targets, out_std):
        err = (preds - targets) * out_std
        return np.stack(
            (
                np.mean(np.square(err), axis=-1),
                np.linalg.norm(err, axis=-1),
                np.abs(err[:, 0]),
                np.abs(err[:, 1]),
                np.abs(err[:, 2]),
            ),
            axis=-1,
        )
