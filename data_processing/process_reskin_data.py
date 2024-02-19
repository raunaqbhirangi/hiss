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
        description="Process demo data: This script allows you to \
            process the raw data into a single h5 file"
    )
    parser.add_argument(
        "--dataset-dir",
        "-dd",
        type=str,
        default="marker_writing_hiss_dataset",
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
    modalities = ["tactile", "kinova"]
    data_suffix = args.data_suffix

    curr_eid = 0
    demo_dirs = get_demo_dirs(args.dataset_dir)
    data_list = []
    freq_lists = {m: [] for m in modalities}
    mod_eids = {m: [0] for m in modalities}
    curr_mod_eid = {m: 0 for m in modalities}

    for demo_dir in tqdm(sorted(demo_dirs)):
        data = {}
        # print(demo_dir)
        for m in modalities:
            try:
                with h5py.File(
                    os.path.join(demo_dir, f"{DATA_FILENAME[m]}"), "r"
                ) as hf:
                    values = np.array(DICT_KEY[m](hf))
                    timestamps = np.array(hf["timestamps"])
                baseline = np.mean(values[:5], axis=0, keepdims=True)
                try:
                    assert timestamps.shape[0] == values.shape[0]
                except AssertionError:
                    clip_len = min(timestamps.shape[0], values.shape[0])
                    timestamps = timestamps[:clip_len]
                    values = values[:clip_len]

                indices = np.arange(len(values))
                if m == "tactile":
                    if np.amax(np.abs(np.diff(values, axis=0))) > 1000.0:
                        # print(demo_dir, np.amax(np.abs(np.diff(values, axis=0))))
                        break
                data[m] = values
                data[f"{m}_timestamps"] = timestamps
                data[f"{m}_indices"] = indices

                freq = 1 / np.mean(np.diff(timestamps))
                freq_lists[m].append(freq)
                if m == "tactile":
                    data[m] -= baseline
            except FileNotFoundError as e:
                print(e)

        if len(data) > 0:
            data_list.append(data)
            curr_mod_eid = {m: curr_mod_eid[m] + len(data[m]) for m in modalities}
            for m in modalities:
                mod_eids[m].append(curr_mod_eid[m])
            done_list = [True] * len(data_list)
    # for m in freq_lists:
    #     print(f"{m}: {np.mean(freq_lists[m])}")

    proc_data_path = get_data_path(args.dataset_dir, data_suffix)
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
