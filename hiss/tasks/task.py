from typing import Dict, List, Tuple
import numpy as np
import numpy.typing as npt
import torch
import h5py


class Task:
    """
    Task object used to store task-specific details such as functions
    for extract input and target tensors, task-specific normalization functions
    task-specific metrics and visualizations
    """

    def __init__(self, name: str) -> None:
        """Initializes Task class.

        Args:
            name:
                name of the task
        """
        self.name = name
        self.metrics = []
        self.loss_fn = torch.nn.MSELoss(reduction="sum")
        self.dset_split = "sequential"

    @staticmethod
    def input_tf_fn(obs_dict: Dict[str, npt.ArrayLike]) -> np.ndarray:
        """
        Extracts input array from a dictionary of arrays, generally a .h5 file
        """
        return np.array(obs_dict["force-state"])

    @staticmethod
    def target_tf_fn(obs_dict: Dict[str, npt.ArrayLike]) -> np.ndarray:
        """
        Extracts target array from a dictionary of arrays, generally a .h5 file
        """
        raise NotImplementedError

    def get_input_target_lists(
        self, data_file: str
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Takes path to .h5 file as input and returns input and target lists of tensors
        """
        input_list, target_list = [], []
        # Extract input and target data using corresponding functions
        with h5py.File(data_file) as hf:
            input_data = torch.Tensor(self.input_tf_fn(hf))
            target_data = torch.Tensor(self.target_tf_fn(hf))
            target_eids = np.array(hf[self.target_eid_key])
            input_eids = np.array(hf[self.input_eid_key])

        # Split data into episodes as lists of tensors
        for ep_i in range(len(target_eids) - 1):
            iid_s, iid_e = input_eids[ep_i], input_eids[ep_i + 1]
            tid_s, tid_e = target_eids[ep_i], target_eids[ep_i + 1]

            curr_input = input_data[iid_s:iid_e]
            if self.filter_input:
                b, a = signal.butter(3, self.filter_cutoff)
                filt_input = signal.filtfilt(b, a, curr_input.numpy(), axis=0)
                curr_input = torch.Tensor(filt_input.copy())
            input_list.append(curr_input)
            if self.predict_diffs:
                target_curr = (
                    target_data[tid_s + 1 : tid_e] - target_data[tid_s : tid_e - 1]
                )
                target_list.append(
                    torch.cat(
                        [torch.zeros((1, target_curr.shape[-1])), target_curr], dim=0
                    )
                )
            else:
                target_curr = target_data[tid_s:tid_e] - target_data[tid_s : tid_s + 1]
                target_list.append(target_curr)

        return input_list, target_list

    def compute_input_shift_scale(self, input_list):
        # Compute shift and scale for input data. Sometimes you might not want this for
        # quantities like quaternion rotations
        input_mean = torch.mean(torch.cat(input_list, dim=0), dim=0)
        input_std = torch.std(torch.cat(input_list, dim=0), dim=0)

        input_std = torch.clamp(input_std, 1e-5)
        return input_mean, input_std

    def compute_target_shift_scale(self, target_list):
        # Compute shift and scale for target data. Sometimes you might not want this for
        # quantities like quaternion rotations
        target_mean = torch.mean(torch.cat(target_list, dim=0), dim=0)
        target_std = torch.std(torch.cat(target_list, dim=0), dim=0)
        target_std = torch.clamp(target_std, 1e-5)
        return target_mean, target_std

    def compute_metrics(self, preds, targets, out_std):
        """
        Compute one or more unnormalized metrics
        """
        return (preds - targets) * out_std
