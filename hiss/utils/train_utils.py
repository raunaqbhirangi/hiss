import os
import numpy as np
import torch
from torch import nn as nn
from torchvision import transforms as transforms

from hiss.dataset import CSPDataset
from hiss.utils import DATA_DIR
import logging
import random


class MaskedLoss(nn.Module):
    def __init__(self, loss_fn, **kwargs) -> None:
        super().__init__(**kwargs)
        self.loss_fn = loss_fn

    def forward(self, inputs, target, lengths):
        pre_mask = torch.arange(inputs.size(1), device=lengths.device).repeat(
            inputs.size(0), 1
        )
        mask = pre_mask < lengths.unsqueeze(1)
        masked_inputs = inputs[mask]
        masked_target = target[mask]
        return self.loss_fn(masked_inputs.squeeze(), masked_target.squeeze())


def get_num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_dsets(
    datafile,
    env_task,
    train_frac=0.8,
    normalize=True,
    val_dset_path=None,
    append_input_deltas=False,
    train_subfrac=1.0,
):
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    dset = CSPDataset.create_from_files(
        datafile,
        task=env_task,
        append_input_deltas=append_input_deltas,
    )

    if normalize:
        data_tf = transforms.Lambda(lambda x: (x - dset.input_mean) / dset.input_std)
        target_tf = transforms.Lambda(
            lambda x: (x - dset.target_mean) / dset.target_std
        )
        dset.add_tf(data_tf, apply_to="input")
        dset.add_tf(target_tf, apply_to="target")

    if hasattr(env_task, "input_tf"):
        dset.add_tf(env_task.input_tf, apply_to="input", prepend=True)
        dset.add_tf(env_task.target_tf, apply_to="target", prepend=True)

    if val_dset_path is not None:
        train_pts = len(dset)
        if env_task.dset_split == "sequential":
            train_dset = torch.utils.data.Subset(
                dset, np.arange(0, int(train_subfrac * train_pts), 1)
            )
        elif env_task.dset_split == "random":
            train_dset, _ = torch.utils.data.random_split(
                dset,
                [int(train_subfrac * train_pts), len(dset) - train_pts],
                generator=torch.Generator().manual_seed(0),
            )
        else:
            raise NotImplementedError
        val_dset = CSPDataset.create_from_files(
            os.path.join(DATA_DIR, val_dset_path),
            task=env_task,
            append_input_deltas=append_input_deltas,
        )
        if normalize:
            val_dset.add_tf(data_tf, apply_to="input")
            val_dset.add_tf(target_tf, apply_to="target")

        # Hack to make compatible with train_frac. Should fix at some point
        log.warning("Using val_dset_path overrides train_frac")
        val_dset = torch.utils.data.Subset(val_dset, np.arange(0, len(val_dset), 1))
    else:
        train_pts = int(train_frac * len(dset))
        if env_task.dset_split == "sequential":
            train_dset = torch.utils.data.Subset(
                dset, np.arange(0, int(train_subfrac * train_pts), 1)
            )
            val_dset = torch.utils.data.Subset(dset, np.arange(train_pts, len(dset), 1))
        elif env_task.dset_split == "random":
            train_dset, val_dset = torch.utils.data.random_split(
                dset,
                [train_pts, len(dset) - train_pts],
                generator=torch.Generator().manual_seed(0),
            )
            if train_subfrac < 1.0:
                train_dset, _ = torch.utils.data.random_split(
                    train_dset,
                    [train_subfrac, 1 - train_subfrac],
                    generator=torch.Generator().manual_seed(0),
                )
        else:
            raise NotImplementedError
    print(f"Dataset lengths: {len(train_dset)}, {len(val_dset)}")

    return train_dset, val_dset
