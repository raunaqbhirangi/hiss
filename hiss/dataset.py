import os
from typing import Callable, List, Optional, Type, Union

import torch
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from hiss.tasks import Task

from hiss.utils import ROOT_DIR


class CSPDataset(Dataset):
    """
    Dataset of a list of torch.Tensor sequences with corresponding targets that
    may themselves be sequences or individual labels
    """

    def __init__(
        self,
        input_list: List[torch.Tensor],
        target_list: List[torch.Tensor],
        maxlen: Optional[int] = None,
        input_norm_fn: Optional[Callable] = None,
        target_norm_fn: Optional[Callable] = None,
        append_input_deltas: bool = False,
    ) -> None:
        """Initializes dataset class.

        Creates a dataset from a list of torch.Tensor sequences, with a corresponding
        list of targets

        Args:
            input_list:
                List of input sequences of tensors
            target_list:
                List of output sequences of tensors.
            maxlen:
                Optional maximum length parameter to trim each sequence to; will
                default to the length of the longest sequence if unspecified
            input_norm_fn:
                Function that takes in the list of input sequences and outputs scale
                and shift parameters
            target_norm_fn:
                Function that takes in the list of target sequences/labels and outputs
                scale and shift parameters
            append_input_deltas:
                Whether to append the difference between consecutive input frames to
                the input data
        """
        super().__init__()
        if append_input_deltas:
            input_diff_list = [in_d[1:] - in_d[:-1] for in_d in input_list]
            input_diff_list = [
                torch.cat((torch.zeros_like(in_d[0:1]), in_d), dim=0)
                for in_d in input_diff_list
            ]
            input_list = [
                torch.cat((in_d, in_d_diff), dim=-1)
                for in_d, in_d_diff in zip(input_list, input_diff_list)
            ]
        self.input_data = pad_sequence(input_list, batch_first=True)
        # TODO: Move this to individual tasks
        self.input_data = torch.clip(self.input_data, -3000.0, 3000.0)
        self.target_data = pad_sequence(target_list, batch_first=True)
        if len(self.target_data.shape) == 2:
            self.target_data = self.target_data.unsqueeze(-1)
        self.input_lens = torch.tensor([len(x) for x in input_list])
        self.target_lens = torch.tensor([len(x) for x in target_list])

        if input_norm_fn is not None:
            self.input_mean, self.input_std = input_norm_fn(input_list)
        if target_norm_fn is not None:
            self.target_mean, self.target_std = target_norm_fn(target_list)

        if maxlen is None:
            self.maxlen = max(self.input_lens)
        else:
            self.maxlen = maxlen
        self.transform = {"input": None, "target": None}

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, index):
        input_data = self.input_data[index]
        target_data = self.target_data[index]
        if self.transform["input"] is not None:
            input_data = self.transform["input"](self.input_data[index])
        if self.transform["target"] is not None:
            target_data = self.transform["target"](self.target_data[index])
        return (
            input_data,
            target_data,
            self.target_lens[index],
        )

    def add_tf(self, tf, apply_to="input", prepend=True):
        assert apply_to in ["input", "target"]
        if self.transform[apply_to] is None:
            self.transform[apply_to] = tf
        else:
            if prepend:
                self.transform[apply_to] = transforms.Compose(
                    [tf, self.transform[apply_to]]
                )
            else:
                self.transform[apply_to] = transforms.Compose(
                    [self.transform[apply_to], tf]
                )

    @staticmethod
    def create_from_files(
        file_paths: Union[str, List[str]],
        task: Type[Task],
        append_input_deltas: bool = False,
        maxlen: Optional[int] = None,
    ) -> "CSPDataset":
        """
        Create a CSPDataset from a set of h5 files and a Task object

        Args:
            file_paths:
                path or list of paths to .h5 files containing data
            task:
                Task object with task-specific data processing functions
            append_input_deltas:
                Whether to append the difference between consecutive input
                frames to the input data
            maxlen:
                Optional maximum length parameter to trim each sequence to; will
                default to the length of the longest sequence if unspecified
        Returns:
            An instance of the CSPDataset class
        """
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        input_list = []
        target_list = []
        for file_path in file_paths:
            curr_path = os.path.join(ROOT_DIR, file_path)
            curr_in, curr_tgt = task.get_input_target_lists(curr_path)
            input_list.extend(curr_in)
            target_list.extend(curr_tgt)

        return CSPDataset(
            input_list,
            target_list,
            input_norm_fn=task.compute_input_shift_scale,
            target_norm_fn=task.compute_target_shift_scale,
            append_input_deltas=append_input_deltas,
            maxlen=maxlen,
        )
