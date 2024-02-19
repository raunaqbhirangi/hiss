import glob
import json
import os
import numpy as np
import h5py
import argparse
import quaternion
from tqdm import tqdm

import sys

sys.path.append("./")

from hiss.utils.data_utils import aggregate_list_of_dicts, get_data_path
from hiss.utils import DATA_DIR


def select_orientation_source(
    data_path, max_ori_error=20.0, grv_only=True, use_ekf=True
):
    """
    Select orientation from one of gyro integration, game rotation vector or EKF orientation.

    Args:
        data_path: path to the compiled data. It should contain "data.hdf5" and "info.json".
        max_ori_error: maximum allow alignment error.
        grv_only: When set to True, only game rotation vector will be used.
                  When set to False:
                     * If game rotation vector's alignment error is smaller than "max_ori_error", use it.
                     * Otherwise, the orientation will be whichever gives lowest alignment error.
                  To force using the best of all sources, set "grv_only" to False and "max_ori_error" to -1.
                  To force using game rotation vector, set "max_ori_error" to any number greater than 360.


    Returns:
        source_name: a string. One of 'gyro_integration', 'game_rv' and 'ekf'.
        ori: the selected orientation.
        ori_error: the end-alignment error of selected orientation.
    """
    ori_names = ["gyro_integration", "game_rv"]
    ori_sources = [None, None, None]

    with open(os.path.join(data_path, "info.json")) as f:
        info = json.load(f)
        ori_errors = np.array(
            [
                info["gyro_integration_error"],
                info["grv_ori_error"],
                info["ekf_ori_error"],
            ]
        )
        init_gyro_bias = np.array(info["imu_init_gyro_bias"])

    with h5py.File(os.path.join(data_path, "data.hdf5")) as f:
        ori_sources[1] = np.copy(f["synced/game_rv"])
        if grv_only or ori_errors[1] < max_ori_error:
            min_id = 1
        else:
            if use_ekf:
                ori_names.append("ekf")
                ori_sources[2] = np.copy(f["pose/ekf_ori"])
            min_id = np.argmin(ori_errors[: len(ori_names)])
            # Only do gyro integration when necessary.
            if min_id == 0:
                ts = f["synced/time"]
                gyro = f["synced/gyro_uncalib"] - init_gyro_bias
                ori_sources[0] = gyro_integration(ts, gyro, ori_sources[1][0])

    return ori_names[min_id], ori_sources[min_id], ori_errors[min_id]


def angular_velocity_to_quaternion_derivative(q, w):
    omega = (
        np.array(
            [
                [0, -w[0], -w[1], -w[2]],
                [w[0], 0, w[2], -w[1]],
                [w[1], -w[2], 0, w[0]],
                [w[2], w[1], -w[0], 0],
            ]
        )
        * 0.5
    )
    return np.dot(omega, q)


def gyro_integration(ts, gyro, init_q):
    """
    Integrate gyro into orientation.
    https://www.lucidar.me/en/quaternions/quaternion-and-gyroscope/
    """
    output_q = np.zeros((gyro.shape[0], 4))
    output_q[0] = init_q
    dts = ts[1:] - ts[:-1]
    for i in range(1, gyro.shape[0]):
        output_q[i] = (
            output_q[i - 1]
            + angular_velocity_to_quaternion_derivative(output_q[i - 1], gyro[i - 1])
            * dts[i - 1]
        )
        output_q[i] /= np.linalg.norm(output_q[i])
    return output_q


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data from the RoNIN Dataset")
    parser.add_argument("-d", "--dataset", choices=["train", "test"], required=True)
    args = parser.parse_args()

    dset_path = os.path.join(DATA_DIR, "ronin")
    data_subset = args.dataset
    modalities = ["gyro", "acce", "tango_pose"]

    proc_data_path = get_data_path("ronin", f"processed_{data_subset}")
    dir_dict = {
        "train": ["train_dataset_1", "train_dataset_2", "seen_subjects_test_set"],
        "test": ["unseen_subjects_test_set"],
    }
    pardir = os.path.abspath(os.path.join(proc_data_path, os.pardir))
    if not os.path.exists(pardir):
        os.makedirs(pardir)

    data_list = []
    freq_lists = {m: [] for m in modalities}
    mod_eids = {m: [0] for m in modalities}
    curr_mod_eid = {m: 0 for m in modalities}
    demo_dirs = []
    for d in dir_dict[data_subset]:
        demo_dirs.extend(glob.glob(os.path.join(dset_path, f"{d}/a*")))
    for ddir in tqdm(demo_dirs):
        data = {}
        with open(os.path.join(ddir, "info.json")) as f:
            info = json.load(f)
        info["ori_source"], ori, info["source_ori_error"] = select_orientation_source(
            ddir, -1, False
        )
        with h5py.File(os.path.join(ddir, "data.hdf5"), "r") as hf:
            gyro_uncalib = hf["synced/gyro_uncalib"]
            acce_uncalib = hf["synced/acce"]
            gyro = gyro_uncalib - np.array(info["imu_init_gyro_bias"])
            acce = np.array(info["imu_acce_scale"]) * (
                acce_uncalib - np.array(info["imu_acce_bias"])
            )
            ts = np.copy(hf["synced/time"])
            tango_pos = np.copy(hf["pose/tango_pos"])
            tango_ori = np.copy(hf["pose/tango_ori"])
            init_tango_ori = quaternion.quaternion(*tango_ori[0])

        # Compute the IMU orientation in the Tango coordinate frame.
        ori_q = quaternion.from_float_array(ori)
        rot_imu_to_tango = quaternion.quaternion(*info["start_calibration"])
        init_rotor = init_tango_ori * rot_imu_to_tango * ori_q[0].conj()
        ori_q = init_rotor * ori_q

        gyro_q = quaternion.from_float_array(
            np.concatenate([np.zeros([gyro.shape[0], 1]), gyro], axis=1)
        )
        acce_q = quaternion.from_float_array(
            np.concatenate([np.zeros([acce.shape[0], 1]), acce], axis=1)
        )
        glob_gyro = quaternion.as_float_array(ori_q * gyro_q * ori_q.conj())[:, 1:]
        glob_acce = quaternion.as_float_array(ori_q * acce_q * ori_q.conj())[:, 1:]

        start_frame = info.get("start_frame", 0)
        ts = ts[start_frame:]
        data["gyro"] = glob_gyro[start_frame:]
        data["acce"] = glob_acce[start_frame:]
        data["tango_pose"] = np.concatenate((tango_pos, tango_ori), axis=-1)[
            start_frame:
        ]
        data["gyro_timestamps"] = ts
        data["acce_timestamps"] = ts
        data["tango_pose_timestamps"] = ts

        data_list.append(data)

        curr_mod_eid = {m: curr_mod_eid[m] + len(data[m]) for m in modalities}
        for m in modalities:
            mod_eids[m].append(curr_mod_eid[m])

    data_list = aggregate_list_of_dicts(data_list)
    for m in modalities:
        data_list[f"{m}_episode_ids"] = mod_eids[m]
    with h5py.File(proc_data_path, "w") as hf:
        for key, value in data_list.items():
            print(key, len(value))
            hf.create_dataset(key, data=value)
