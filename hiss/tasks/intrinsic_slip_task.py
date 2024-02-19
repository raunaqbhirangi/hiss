import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import h5py
from scipy import signal

from hiss.tasks.task import Task


class IntrinsicSlipTask(Task):
    def __init__(
        self, filter_input=False, dset_split="sequential", filter_cutoff=0.03
    ) -> None:
        super().__init__("IntrinsicSlipTask")
        self.filter_input = filter_input
        self.metrics = [
            "mse",
            "mse_xy",
            "mse_th",
            "err",
            "err_x",
            "err_y",
            "err_theta",
            "cum_err_xy",
            "cum_err_th",
        ]
        self.dset_split = dset_split
        self.filter_cutoff = filter_cutoff
        self.input_eid_key = "tactile_episode_ids"
        self.target_eid_key = "kinova_episode_ids"
        self.predict_diffs = True

    def input_tf_fn(self, obs_dict):
        return obs_dict["tactile"]

    def target_tf_fn(self, obs_dict):
        return obs_dict["kinova"]

    def _tf_traj(self, traj):
        init_ori = R.from_quat(traj[0, 3:])
        init_tf_mat = np.eye(4)
        init_tf_mat[:3, :3] = init_ori.as_matrix()
        init_tf_mat[:3, 3] = traj[0, :3]
        init_tf_mat_inv = np.linalg.inv(init_tf_mat)
        kinova_mats = np.zeros((len(traj), 4, 4))
        kinova_mats[:, 3, 3] = 1.0
        # tf_z = (init_tf_mat_inv @ np.array([0, 0, 1, 0]))[:3]
        # tf_x = (init_tf_mat_inv @ np.array([1, 0, 0, 0]))[:3]

        kinova_mats[:, :3, :3] = R.from_quat(traj[:, 3:]).as_matrix()
        kinova_mats[:, :3, 3] = traj[:, :3]
        kinova_mats = np.matmul(kinova_mats, init_tf_mat_inv)
        tf_traj = np.zeros((traj.shape[0], 4))
        tf_traj[:, :3] = traj[:, :3] - traj[0, :3]
        tf_traj[:, 3] = R.from_matrix(kinova_mats[:, :3, :3]).as_rotvec()[:, 1]

        return tf_traj

    def get_input_target_lists(self, data_file: str):
        """
        Takes path to .h5 file as input and returns input and target lists of tensors
        """
        input_list, target_list = [], []
        # Extract input and target data using corresponding functions
        with h5py.File(data_file) as hf:
            input_data = np.array(self.input_tf_fn(hf))
            target_data = np.array(self.target_tf_fn(hf))
            target_eids = np.array(hf[self.target_eid_key])
            input_eids = np.array(hf[self.input_eid_key])

        # Split data into episodes as lists of tensors
        for ep_i in range(len(target_eids) - 1):
            iid_s, iid_e = input_eids[ep_i], input_eids[ep_i + 1]
            tid_s, tid_e = target_eids[ep_i], target_eids[ep_i + 1]

            curr_input = torch.from_numpy(input_data[iid_s:iid_e])
            if self.filter_input:
                b, a = signal.butter(3, self.filter_cutoff)
                filt_input = signal.filtfilt(b, a, curr_input, axis=0)
                curr_input = torch.Tensor(filt_input.copy())
            input_list.append(curr_input)
            target_curr = self._tf_traj(target_data[tid_s:tid_e])[:, [0, 2, 3]]
            target_curr_diff = torch.from_numpy(target_curr[1:] - target_curr[:-1])
            target_list.append(
                torch.cat([torch.zeros((1, 3)), target_curr_diff], dim=0)
            )
        return input_list, target_list

    def compute_input_shift_scale(self, input_list):
        input_mean = 0.0
        input_min = torch.amin(torch.cat(input_list, dim=0), dim=0)
        input_max = torch.amax(torch.cat(input_list, dim=0), dim=0)
        input_std = (input_max - input_min) / 5
        return input_mean, input_std

    def compute_target_shift_scale(self, target_list):
        target_mean, target_std = super().compute_target_shift_scale(target_list)
        target_std[1] = target_std[0]
        return torch.zeros((3,)), target_std

    def compute_metrics(self, preds, targets, out_std):
        err = (preds - targets) * out_std
        cum_err = np.cumsum(err, axis=0)
        return np.stack(
            (
                np.mean(np.square(err), axis=-1),
                np.mean(np.square(err[:, :-1]), axis=-1),
                np.square(err[:, -1]),
                np.linalg.norm(err, axis=-1),
                np.abs(err[:, 0]),
                np.abs(err[:, 1]),
                np.abs(err[:, 2]),
                np.linalg.norm(cum_err[:, :2], axis=-1),
                cum_err[:, 2],
            ),
            axis=-1,
        )
