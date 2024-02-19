import glob
import os

import numpy as np

from hiss.utils import DATA_DIR


def aggregate_list_of_dicts(list_of_dicts):
    """
    Given a list of dicts, return a single dict with the same keys, but with
    values that are lists of the values in the original dicts.
    """
    if len(list_of_dicts) == 0:
        return {}
    if isinstance(list_of_dicts[0], list):
        list_of_dicts = [aggregate_list_of_dicts(ld) for ld in list_of_dicts]

    keys = list_of_dicts[0].keys()
    result = {k: [] for k in keys}
    for d in list_of_dicts:
        for k in keys:
            result[k].append(d[k])
    for k in keys:
        try:
            result[k] = np.array(result[k])
        except ValueError:
            try:
                result[k] = np.concatenate(result[k], axis=0)
            except ValueError:
                print(f"Returning lists instead of np-arrays for key {k}")
    return result


def get_data_path(dataset_dir, data_suffix):
    data_dir = os.path.join(DATA_DIR, dataset_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    data_path = os.path.join(data_dir, f"{dataset_dir}_{data_suffix}.h5")
    return data_path


def get_demo_dirs(dataset_dir):
    if dataset_dir.startswith("intrinsic_slip"):
        demo_dirs = sorted(glob.glob(f"{DATA_DIR}/{dataset_dir}/*/demonstration_*"))
    else:
        demo_dirs = sorted(glob.glob(f"{DATA_DIR}/{dataset_dir}/demonstration_*"))
    return demo_dirs


DATA_FILENAME = {
    "tactile": "reskin_sensor_values.h5",
    "xela": "touch_sensor_values.h5",
    "kinova": "kinova_cartesian_states.h5",
    "extreme3d": "extreme3d_values.h5",
}


def flattened_xela(d):
    return np.concatenate(
        (
            np.array(d["finger_sensor_values"]).reshape(
                d["finger_sensor_values"].shape[0], -1
            ),
            np.array(d["fingertip_sensor_values"]).reshape(
                d["fingertip_sensor_values"].shape[0], -1
            ),
            np.array(d["palm_sensor_values"]).reshape(
                d["palm_sensor_values"].shape[0], -1
            ),
        ),
        axis=-1,
    )


DICT_KEY = {
    "xela": lambda d: flattened_xela(d),
    "gripper": lambda d: d["widths"],
    "kinova": lambda d: np.concatenate((d["positions"], d["orientations"]), axis=-1),
    "extreme3d": lambda d: np.concatenate((d["axes"], d["buttons"]), axis=-1),
    "tactile": lambda d: d["sensor_values"],
}
