import h5py
import os
import argparse

import numpy as np
from tqdm import tqdm

import sys

sys.path.append("./")

from hiss.utils.data_utils import (
    aggregate_list_of_dicts,
    get_data_path,
    get_demo_dirs,
)
from hiss.utils.data_utils import DATA_FILENAME, DICT_KEY


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process demo data: This script allows you to process the \
            raw data into a single h5 file"
    )
    parser.add_argument(
        "--dataset-dir",
        "-dd",
        type=str,
        default="joystick_control_hiss_dataset",
        help="Directory containing the dataset to process",
    )
    parser.add_argument(
        "--data-suffix",
        "-ds",
        type=str,
        default="processed",
        help="Suffix to append to the processed data file",
    )
    args = parser.parse_args()

    # List modalities to process.
    modalities = ["xela", "extreme3d"]
    demo_dirs = get_demo_dirs(args.dataset_dir)
    proc_data_path = get_data_path(args.dataset_dir, args.data_suffix)

    dur_list = []
    data_list = []
    freq_lists = {m: [] for m in modalities}
    mod_eids = {m: [0] for m in modalities}
    curr_mod_eid = {m: 0 for m in modalities}

    for demo_dir in tqdm(sorted(demo_dirs)):
        data = {}
        dur_curr = []
        for m in modalities:
            try:
                with h5py.File(
                    os.path.join(demo_dir, f"{DATA_FILENAME[m]}"), "r"
                ) as hf:
                    values = np.array(DICT_KEY[m](hf))
                    timestamps = np.array(hf["timestamps"])
                if m == "xela":
                    baseline = np.median(values[:100], axis=0, keepdims=True)
                try:
                    assert timestamps.shape[0] == values.shape[0]
                except AssertionError:
                    clip_len = min(timestamps.shape[0], values.shape[0])
                    timestamps = timestamps[:clip_len]
                    values = values[:clip_len]

                indices = np.arange(len(values))
                data[m] = values
                data[f"{m}_timestamps"] = timestamps
                data[f"{m}_indices"] = indices

                freq = 1 / np.mean(np.diff(timestamps))
                freq_lists[m].append(freq)
                if m == "xela":
                    data[m] -= baseline
                    data[m] = np.clip(data[m], -1000, 1000)

                dur_curr.append(np.around(timestamps[-1] - timestamps[0]))
                if m == "extreme3d":
                    if min(dur_curr) < 15.0 or max(dur_curr) > 90.0:
                        # print(demo_dir, dur_curr)
                        data = {}
                        break
                    dur_list.append(dur_curr)

            except FileNotFoundError as e:
                print(e)
                # break

        if len(data) > 0:
            data_list.append(data)

            curr_mod_eid = {m: curr_mod_eid[m] + len(data[m]) for m in modalities}
            for m in modalities:
                mod_eids[m].append(curr_mod_eid[m])
            done_list = [True] * len(data_list)
    for m in freq_lists:
        print(f"{m}: {np.mean(freq_lists[m])}")
    # dur_list = np.array(dur_list)
    pardir = os.path.abspath(os.path.join(proc_data_path, os.pardir))
    if not os.path.exists(pardir):
        os.makedirs(pardir)
    data_list = aggregate_list_of_dicts(data_list)
    for m in modalities:
        data_list[f"{m}_episode_ids"] = mod_eids[m]
    with h5py.File(proc_data_path, "w") as hf:
        for key, value in data_list.items():
            print(key, len(value))
            hf.create_dataset(key, data=value)
