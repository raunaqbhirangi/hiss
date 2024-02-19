import os
import numpy as np
import h5py
from tqdm import tqdm

import sys

sys.path.append("./")

from hiss.utils.data_utils import aggregate_list_of_dicts, get_data_path
from hiss.utils import DATA_DIR
from scipy.spatial.transform import Rotation as R


def pose_to_transformation_matrix(pose):
    tx, ty, tz, qx, qy, qz, qw = pose
    transformation_matrix = np.eye(4)
    transformation_matrix[0:3, 3] = [tx, ty, tz]
    rotation_matrix = R.from_quat([qx, qy, qz, qw]).as_matrix()
    transformation_matrix[0:3, 0:3] = rotation_matrix

    return transformation_matrix


def calculate_gt_in_B0(T_b_in_w, T_w_to_B0):
    T_b_in_B0 = np.dot(T_w_to_B0, T_b_in_w)
    translation = T_b_in_B0[0:3, 3]
    rotation_matrix = T_b_in_B0[0:3, 0:3]

    quaternion = R.from_matrix(rotation_matrix).as_quat()
    normalized_quaternion = R.from_quat(quaternion).as_quat()

    return np.concatenate((translation, normalized_quaternion))


if __name__ == "__main__":
    plot_data = False
    dset_path = os.path.join(DATA_DIR, "vector")

    modalities = ["gyro", "acce", "magn", "pose"]

    proc_data_path = get_data_path("vector", f"processed")

    pardir = os.path.abspath(os.path.join(proc_data_path, os.pardir))
    if not os.path.exists(pardir):
        os.makedirs(pardir)

    data_list = []
    freq_lists = {m: [] for m in modalities}
    mod_eids = {m: [0] for m in modalities}
    curr_mod_eid = {m: 0 for m in modalities}

    demo_dirs = [
        d for d in os.listdir(dset_path) if os.path.isdir(os.path.join(dset_path, d))
    ]
    for ddir in tqdm(demo_dirs):
        data = {}
        imu_ts = []
        gyro = []
        acce = []
        magn = []

        pose_in_w = []
        pose_in_B0 = []
        gt_ts = []

        imu_path = os.path.join(dset_path, ddir, "imu.txt")

        with open(imu_path, "r") as file:
            for _ in range(3):
                next(file)
            columns = []
            for line in file:
                parts = line.strip().split(" ")  # Splitting by comma for CSV format
                if not columns:
                    # Initialize lists to hold each column's data
                    columns = [[] for _ in range(len(parts))]
                for i, part in enumerate(parts):
                    columns[i].append(float(part))

            imu_ts = columns[0]
            gyro = columns[1:4]
            acce = columns[4:7]
            magn = columns[7:11]

        gt_path = os.path.join(dset_path, ddir, "gt.txt")
        with open(gt_path, "r") as file:
            for _ in range(2):
                next(file)
            columns = []
            for line in file:
                parts = line.strip().split(" ")  # Splitting by comma for CSV format
                if not columns:
                    # Initialize lists to hold each column's data
                    columns = [[] for _ in range(len(parts))]
                for i, part in enumerate(parts):
                    columns[i].append(float(part))

            gt_ts = columns[0]
            pose_in_w = np.array(columns[1:]).T

            T_Bo_in_w = pose_to_transformation_matrix(pose_in_w[0])

            T_w_in_B0 = np.linalg.inv(T_Bo_in_w)

            for i in range(0, len(pose_in_w)):
                pose_in_B0.append(
                    calculate_gt_in_B0(
                        pose_to_transformation_matrix(pose_in_w[i]), T_w_in_B0
                    )
                )

        data["gyro"] = np.array(gyro).T
        data["acce"] = np.array(acce).T
        data["magn"] = np.array(magn).T
        data["pose"] = np.array(pose_in_B0)

        px = np.array(pose_in_B0)[:, 0]
        py = np.array(pose_in_B0)[:, 1]
        pz = np.array(pose_in_B0)[:, 2]

        data["gyro_timestamps"] = np.array(imu_ts)
        data["acce_timestamps"] = np.array(imu_ts)
        data["magn_timestamps"] = np.array(imu_ts)
        data["pose_timestamps"] = np.array(gt_ts)

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
