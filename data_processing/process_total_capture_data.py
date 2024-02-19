import os
import numpy as np
import argparse
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


def parse_sensors_file(file_path, name_list):
    with open(file_path, "r") as file:
        num_sensors, num_frames = map(int, file.readline().split())
        sensor_data = {f"rot_{name}": [] for name in name_list}
        sensor_data = {f"gyro_{name}": [] for name in name_list}
        sensor_data.update({f"acce_{name}": [] for name in name_list})
        sensor_data.update({f"magn_{name}": [] for name in name_list})

        for _ in range(num_frames):
            frame_number = int(file.readline().strip())
            for _ in range(num_sensors):
                parts = file.readline().strip().split()
                sensor_name = parts[0]

                rot_data = list(map(float, parts[1:5]))
                acce_data = list(map(float, parts[5:8]))
                gyro_data = list(map(float, parts[8:11]))
                magn_data = list(map(float, parts[11:14]))

                sensor_data[f"rot_{sensor_name}"].append(rot_data)
                sensor_data[f"gyro_{sensor_name}"].append(gyro_data)
                sensor_data[f"acce_{sensor_name}"].append(acce_data)
                sensor_data[f"magn_{sensor_name}"].append(magn_data)

    for key in sensor_data:
        sensor_data[key] = np.array(sensor_data[key])

    return sensor_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data from the RoNIN Dataset")
    parser.add_argument("-d", "--dataset", choices=["train", "test"], required=True)
    args = parser.parse_args()
    dataset_dir = "total_capture"
    if args.dataset == "test":
        dataset_dir = "total_capture_test"
    dset_path = os.path.join(DATA_DIR, dataset_dir)
    proc_data_path = get_data_path(dataset_dir, f"processed")

    imu_modalities_name = [
        "gyro",
        "acce",
        "rot_inertial_frame",
        "rot_global_frame",
        "acce_inertial_frame",
        "acce_global_frame",
        "magn",
    ]
    pose_modalities_name = ["position", "orientation", "pose"]
    imu_name_list = [
        "Head",
        "Sternum",
        "Pelvis",
        "L_UpArm",
        "R_UpArm",
        "L_LowArm",
        "R_LowArm",
        "L_UpLeg",
        "R_UpLeg",
        "L_LowLeg",
        "R_LowLeg",
        "L_Foot",
        "R_Foot",
    ]
    joint_name_list = [
        "Hips",
        "Spine",
        "Spine1",
        "Spine2",
        "Spine3",
        "Neck",
        "Head",
        "RightShoulder",
        "RightArm",
        "RightForeArm",
        "RightHand",
        "LeftShoulder",
        "LeftArm",
        "LeftForeArm",
        "LeftHand",
        "RightUpLeg",
        "RightLeg",
        "RightFoot",
        "LeftUpLeg",
        "LeftLeg",
        "LeftFoot",
    ]

    imu_modalities = [
        f"{modality}_{name}"
        for modality in imu_modalities_name
        for name in imu_name_list
    ]
    pose_modalities = [
        f"{modality}_{name}"
        for modality in pose_modalities_name
        for name in joint_name_list
    ]

    modalities = imu_modalities + pose_modalities

    sensor_frequncy = 60

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
        cali_i_to_w = {}
        imu_ts = []
        gyro = []
        acce = []
        magn = []

        pose_in_w = []
        pose_in_B0 = []
        gt_ts = []

        cali_path = os.path.join(dset_path, ddir, "calib_imu_ref.txt")

        with open(cali_path, "r") as file:
            for _ in range(1):
                next(file)
            for line in file:
                values = line.strip().split()
                cali_i_to_w[values[0]] = [
                    float(values[1]),
                    float(values[2]),
                    float(values[3]),
                    float(values[4]),
                ]

        imu_path = os.path.join(dset_path, ddir, "imu.sensors")

        with open(imu_path, "r") as file:
            num_sensors, num_frames = map(int, file.readline().split())
            sensor_data = {f"{name}": [] for name in imu_modalities}
            interval = 1 / sensor_frequncy
            timestamps = interval * np.arange(num_frames)
            for imu_name in imu_name_list:
                data[f"rot_global_frame_{imu_name}_timestamps"] = timestamps
                data[f"acce_global_frame_{imu_name}_timestamps"] = timestamps
            for joint_name in joint_name_list:
                data[f"pose_{joint_name}_timestamps"] = timestamps
            data["timestamps"] = timestamps

            for _ in range(num_frames):
                frame_number = int(file.readline().strip())
                for _ in range(num_sensors):
                    parts = file.readline().strip().split()
                    sensor_name = parts[0]

                    rot_data = list(map(float, parts[2:5] + parts[1:2]))
                    acce_data = list(map(float, parts[5:8]))
                    gyro_data = list(map(float, parts[8:11]))
                    magn_data = list(map(float, parts[11:14]))

                    rot_inertial_to_global = cali_i_to_w[sensor_name]
                    R_imu_in_i = R.from_quat(rot_data).as_matrix()
                    R_imu_in_g = (
                        R.from_quat(rot_inertial_to_global).as_matrix().dot(R_imu_in_i)
                    )

                    rot_imu_in_global = R.from_matrix(R_imu_in_g).as_quat()

                    acce_inertial_data = list(R_imu_in_i.dot(np.array(acce_data)))
                    acce_global_data = list(R_imu_in_g.dot(np.array(acce_data)))

                    sensor_data[f"rot_inertial_frame_{sensor_name}"].append(rot_data)
                    sensor_data[f"rot_global_frame_{sensor_name}"].append(
                        rot_imu_in_global
                    )
                    sensor_data[f"gyro_{sensor_name}"].append(gyro_data)
                    sensor_data[f"acce_{sensor_name}"].append(acce_data)
                    sensor_data[f"acce_inertial_frame_{sensor_name}"].append(
                        acce_inertial_data
                    )
                    sensor_data[f"acce_global_frame_{sensor_name}"].append(
                        acce_global_data
                    )
                    sensor_data[f"magn_{sensor_name}"].append(magn_data)

        for key in sensor_data:
            sensor_data[key] = np.array(sensor_data[key])

        position_path = os.path.join(dset_path, ddir, "gt_skel_gbl_pos.txt")
        orientation_path = os.path.join(dset_path, ddir, "gt_skel_gbl_ori.txt")

        pose_data = {f"{name}": [] for name in pose_modalities}
        with open(position_path, "r") as file:
            for _ in range(1):
                next(file)
            for lid, line in enumerate(file):
                values = line.strip().split()
                if lid == num_frames:  # remove the last frame
                    break
                for i, part in enumerate(joint_name_list):
                    key = f"position_{part}"

                    pose_data[key].append(
                        [
                            float(values[i * 3]),
                            float(values[i * 3 + 1]),
                            float(values[i * 3 + 2]),
                        ]
                    )

        with open(orientation_path, "r") as file:
            for _ in range(1):
                next(file)
            for lid, line in enumerate(file):
                values = line.strip().split()
                if lid == num_frames:  # remove the last frame
                    break
                for i, part in enumerate(joint_name_list):
                    key = f"orientation_{part}"
                    pose_data[key].append(
                        [
                            float(values[i * 4 + 1]),
                            float(values[i * 4 + 2]),
                            float(values[i * 4 + 3]),
                            float(values[i * 4]),
                        ]
                    )

        for joint in joint_name_list:
            orientation_key = f"orientation_{joint}"
            position_key = f"position_{joint}"
            pose_key = f"pose_{joint}"
            pose_in_B0 = []

            ori = np.array(pose_data[orientation_key])
            position = np.array(pose_data[position_key])
            pose_in_w = np.concatenate((position, ori), axis=1)

            T_Bo_in_w = pose_to_transformation_matrix(pose_in_w[0])
            T_w_in_B0 = np.linalg.inv(T_Bo_in_w)

            for i in range(0, len(pose_in_w)):
                pose_in_B0.append(
                    calculate_gt_in_B0(
                        pose_to_transformation_matrix(pose_in_w[i]), T_w_in_B0
                    )
                )

            pose_data[pose_key] = np.array(pose_in_B0)

            px = np.array(pose_in_B0)[:, 0]
            py = np.array(pose_in_B0)[:, 1]
            pz = np.array(pose_in_B0)[:, 2]

        data.update(pose_data)
        data.update(sensor_data)

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
