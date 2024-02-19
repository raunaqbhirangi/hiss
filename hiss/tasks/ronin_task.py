import numpy as np
import torch
from hiss.tasks.task import Task


def change_cf(ori, vectors):
    """
    Euler-Rodrigous formula v'=v+2s(rxv)+2rx(rxv)
    :param ori: quaternion [n]x4
    :param vectors: vector nx3
    :return: rotated vector nx3
    """
    assert ori.shape[-1] == 4
    assert vectors.shape[-1] == 3

    if len(ori.shape) == 1:
        ori = torch.from_numpy(np.repeat([ori], vectors.shape[0], axis=0)).float()
    q_s = ori[:, :1]
    q_r = ori[:, 1:]
    tmp = torch.cross(q_r, vectors)
    vectors = torch.add(
        torch.add(vectors, torch.multiply(2, torch.multiply(q_s, tmp))),
        torch.multiply(2, torch.cross(q_r, tmp)),
    )
    return vectors


class RotateSeqTf:
    def __init__(self) -> None:
        self.rng = np.random.default_rng(seed=0)

    def __call__(self, sample):
        # if sample.shape[-1] == self.input_dim:
        #     self.random_angle = np.random.random() * 2 * np.math.pi
        rangle = self.rng.random() * 2 * np.math.pi
        tf = np.array([np.cos(rangle), 0, 0, np.sin(rangle)])
        for i in range(0, sample.shape[-1], 3):
            vector = sample[:, i : i + 3]
            if vector.shape[-1] == 2:
                vector = torch.cat((vector, torch.zeros((sample.shape[0], 1))), dim=-1)
                sample[:, i : i + 3] = change_cf(tf, vector)[:, : sample.shape[-1]]
            else:
                sample[:, i : i + 3] = change_cf(tf, vector)

        return sample


class RoninTask(Task):
    def __init__(self, filter_input=False, use_tfs=False, filter_cutoff=0.03) -> None:
        super().__init__("RoninTask")
        self.metrics = ["mse", "err", "err_x", "err_y", "cum_err"]
        self.filter_input = filter_input
        if use_tfs:
            self.input_tf = RotateSeqTf()
            self.target_tf = RotateSeqTf()
        self.filter_cutoff = filter_cutoff
        self.input_eid_key = "gyro_episode_ids"
        self.target_eid_key = "tango_pose_episode_ids"
        self.predict_diffs = True

    def input_tf_fn(self, obs_dict):
        acc_data = obs_dict["acce"]
        gyro_data = obs_dict["gyro"]
        return np.concatenate((acc_data, gyro_data), axis=-1)

    def target_tf_fn(self, obs_dict):
        tango_pos = obs_dict["tango_pose"][:, :3]
        tango_quat = obs_dict["tango_pose"][:, 3:]
        return tango_pos[:, :2]

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
