# CSP-Bench:
<p align="center">
    <img src="https://github.com/raunaqbhirangi/hiss/assets/73357354/20eec6a6-4e65-435b-a80e-7d9905ee03f1" style="width: 500px; height: auto; ">
</p>

## About
CSP-Bench is a continuous sequence-to-sequence prediction benchmark consisting of six datasets. Three of these datasets were [collected](https://github.com/raunaqbhirangi/vt_state/tree/release/data_processing#collected-datasets) in-house (Marker Writing, Intrinsic Slip, Joystick Control) while three others were [curated](https://github.com/raunaqbhirangi/vt_state/tree/release/data_processing#curated-imu-datasets) ([RoNIN](https://ronin.cs.sfu.ca), [VECtor](https://star-datasets.github.io/vector/), [TotalCapture](https://cvssp.org/data/totalcapture/)) from existing literature. Here, we release documentation for all collected data as well as preprocessing steps and scripts for the curated data.

## Collected Datasets
In addition to the modalities used for defining the sequence-to-sequence prediction task, we collect data from a number of additional synchronized modalities, such as camera feeds. To streamline the process of reproducing results presented in the accompanying paper, we release two versions of the dataset -- a compact HiSS version consisting of only the modalities used for training HiSS models, and a full version consisting of data collected from all modalities. All three collected datasets can be downloaded [here](https://drive.google.com/drive/folders/1YzhnRvfdEq_Y_GatlorA-SCWMH15hux8?usp=sharing).


### HiSS Version

This version consists of the tactile sensor data used as input to the models and the robot/joystick data used for supervision. If you're only interested in the sequential prediction tasks described in the paper, download this version.

```txt
marker_writing_hiss_dataset or intrinsic_slip_hiss_dataset
|
└── demonstration_<id>
    ├── kinova_cartesian_states.h5
    └── reskin_sensor_values.h5
```

```txt
joystick_control_hiss_dataset
|
└── demonstration_<id>
    ├── extreme3d_values.h5
    └── touch_sensor_values.h5
```


### Full Version

This version includes the camera feed and other modalities not present in the HiSS version. If you're only interested in modalities and tasks in addition to the sequential prediction tasks described in the paper, download this version.

```txt
marker_writing_full_dataset or intrinsic_slip_full_dataset
|
├── demostration_<id>
│   ├── cam_0_depth.h5
│   ├── cam_0_rgb_video.avi
│   ├── cam_0_rgb_video.metadata
│   ├── cam_1_depth.h5
│   ├── cam_1_rgb_video.avi
│   ├── cam_1_rgb_video.metadata
│   ├── kinova_cartesian_states.h5
│   ├── kinova_joint_states.h5
│   ├── onrobot_commanded_joint_states.h5
│   ├── onrobot_joint_states.h5
│   └── reskin_sensor_values.h5
│
└── missing_file.txt
```

```txt
joystick_control_full_dataset
|
├── demostration_<id>
│   ├── allegro_commanded_joint_states.h5
│   ├── allegro_fingertip_states.h5
│   ├── allegro_joint_states.h5
│   ├── extreme3d_values.h5
│   ├── franka_cartesian_states.h5
│   ├── franka_joint_states.h5
│   ├── cam_0_depth.h5
│   ├── cam_0_rgb_video.avi
│   ├── cam_0_rgb_video.metadata
│   ├── cam_2_depth.h5
│   ├── cam_2_rgb_video.avi
│   ├── cam_2_rgb_video.metadata
│   └── touch_sensor_values.h5
│
├── 3_camera_list.txt (includes all the folders with 1 extra camera)
└── missing_file.txt
```

### Description of Files

- `<demonstration_<id>>`

    Each demonstration folder contains a set of files related to a single sequence of robotic operations.

    - **`\*.h5` files**: HDF5 files containing various sensor data.
    - **`\*.avi` files**: Video files capturing the RGB feed from the RealSense cameras.
    - **`\*.metadata` files**: Metadata files associated with the corresponding RGB video files.

​
### Auxiliary files
**missing_file.txt**: Due to hardware failures, some trajectories in these datasets are missing data from certain modalities. None of these missing files correspond to the HiSS version which is available in its entirety. This file keeps a record of all the missing files in the corresponding dataset; each line of the file corresponds to a specific demonstration and the files missing for that demonstration:

```txt
demonstration_<id> <missing_file1> <missing_file2> ...
```

**3_camera_list.txt**: For `joystick_control_full_dataset` we have 135 sequences that have an additional camera view. This file enumerates all such demonstrations that include data from an additional camera.

### h5 file contents
We also document the contents of each of the h5 files present in both versions of the datasets.

#### kinova_cartesian_states.h5

```txt
kinova_cartesian_states.h5
|
├── file_name
├── num_datapoints
├── orientations (orientation of the end effector in Quaternion: x,y,z,w)
├── positions (position of the end effector: x,y,z in meters)
├── record_duration (length of this sequence in seconds)
├── record_end_time (end timestamp of this sequence in seconds)
├── record_start_time (start timestamp of this sequence in seconds)
├── record_frequency (frequency of Kinova readings in Hz)
└── timestamps (timestamps of readings in seconds )
```

#### reskin_sensor_values.h5

```txt
reskin_sensor_values.h5
|
├── file_name
├── num_datapoints
├── record_duration (length of this sequence in seconds)
├── record_end_time (end timestamp of this sequence in seconds)
├── record_start_time (start timestamp of this sequence in seconds)
├── record_frequency (frequency of Kinova readings in Hz)
├── sensor_values (ReSkin readings [bx1 by1, bz1, ......, bx10, by10, bz10])
└── timestamps (timestamps of readings in seconds )
```

#### extreme3d_values.h5

```txt
extreme3d_values.h5
|
├── file_name
├── num_datapoints
├── record_duration (length of this sequence in seconds)
├── record_end_time (end timestamp of this sequence in seconds)
├── record_start_time (start timestamp of this sequence in seconds)
├── record_frequency (frequency of Kinova readings in Hz)
├── axes (Joystick readings [x, y, z-twist])
├── buttons
├── hat_switch
├── throttle
└── timestamps (timestamps of readings in seconds )
```

# Curated Datasets

### RoNIN

https://www.frdr-dfdr.ca/repo/dataset/816d1e8c-1fc3-47ff-b8ea-a36ff51d682a



#### Preprocess:

##### 1. Extract the dataset

The dataset structure should look like:

```txt
ronin
|
└── <sequence_name>
    ├── data.hdf5
    └── info.json
```



### VECtor

https://star-datasets.github.io/vector/download/



#### Preprocess:
##### 1. Please download txt files for <ins>imu</ins> and <ins>ground truth</ins> for each sequence to <DATA_DIR>/vector.

##### 2. Reformat the dataset structure
```
python dataset_processing/vector_reformat.py
```

The dataset structure should look like:

```txt
vector
|
└── <sequence_name>
    ├── gt.txt
    └── imu.txt
```



### TotalCapture

https://cvssp.org/data/totalcapture/data/

Please note that the unit of groundtruth is **inches**

This dataset provides 12 IMU sensors and the ground truth poses of 21 joints.

```txt
imu_name_list = ['Head', 'Sternum', 'Pelvis', 'L_UpArm', 'R_UpArm', 'L_LowArm', 'R_LowArm', 'L_UpLeg', 'R_UpLeg', 'L_LowLeg', 'R_LowLeg', 'L_Foot', 'R_Foot']

joint_name_list = ['Hips', 'Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Head', 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand', 'RightUpLeg', 'RightLeg', 'RightFoot', 'LeftUpLeg', 'LeftLeg', 'LeftFoot']
```

#### Preprocess:

##### 1. Download <ins>IMU</ins> and <ins>Vicon Groundtruth (real-world position and orientation)</ins> for each Subject to <DATA_DIR>/total_capture.

##### 2. Reformat the dataset structure.

```
python dataset_processing/total_capture_reformat.py
```

The dataset structure should look like:

```txt
total_capture
|
└── <Subjet_id>_<sequence_name>
    ├── calib_imu_ref.txt (renamed from <Subjet_id>_<sequence_name>_calib_imu_ref.txt)
    ├── gt_skel_gbl_ori.txt
    ├── gt_skel_gbl_pos.txt
    └── imu.txt (renamed from <sequence_name>_Xsens_AuxFields.sensors)
```
