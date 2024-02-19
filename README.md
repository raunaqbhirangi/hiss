# Hierarchical State Space Models (HiSS)
![poster-compressed](https://github.com/raunaqbhirangi/hiss/assets/73357354/33fe0d1d-a1f2-480b-9d5b-ac8318fbbae4)


> __Hierarchical State Space Models for Continuous Sequence-to-Sequence Modeling__ \
> Raunaq Bhirangi, Chenyu Wang, Venkatesh Pattabiraman, Carmel Majidi, Abhinav Gupta, Tess Hellebrekers and Lerrel Pinto\
> Paper: https://arxiv.org/abs/2402.10211 \
> Website: https://hiss-csp.github.io/

## About
HiSS is a simple technique that stacks deep state space models like [S4]() and [Mamba]() to reason over continuous sequences of sensory data over mutiple temporal hierarchies. We also release CSP-Bench: a benchmark for sequence-to-sequence prediction from sensory data.

## Installation
1. Clone the repository

2. Create a conde environment from the provided `env.yml` file: ```conda env create -f env.yml```

3. Install Mamba based on the official [instructions](https://github.com/state-spaces/mamba/tree/main?tab=readme-ov-file#installation).

Note: If you run into CUDA issues while installing Mamba, run ```export CUDA_HOME=$CONDA_PREFIX```, and try again. If you still have problems, install both `causal_conv1d` and `mamba-ssm` from source.

## Data processing
1. Refer to [data_processing/README](./data_processing/README.md) to download and extract the required dataset.

2. Set the `DATA_DIR` variable in the [`hiss/utils/__init__.py`](https://github.com/raunaqbhirangi/hiss/blob/main/hiss/utils/__init__.py) file. This is the path to the parent directory which contains folders corresponding to every dataset.

3. Process the datasets into format compatible with training
<br>__Marker Writing__: `python data_processing/process_reskin_data.py -dd marker_writing_<hiss/full>_dataset`
<br>__Intrinsic Slip__: `python data_processing/process_reskin_data.py -dd intrinsic_slip_<hiss/full>_dataset`
<br>__Joystick Control__: `python data_processing/process_xela_data.py -dd joystick_control_<hiss/full>_dataset`
<br>__RoNIN__: `python data_processing/process_ronin_data.py`
<br>__VECtor__: `python data_processing/process_vector_data.py`
<br>__TotalCapture__: `python data_processing/process_total_capture_data.py`

5. Run `create_dataset.py` for the respective dataset to preprocess data and resample it at the desired frequencies.
<br>__Marker Writing__: `python create_dataset.py --config-name marker_writing_config`
<br>__Intrinsic Slip__: `python create_dataset.py --config-name intrinsic_slip_config`
<br>__Joystick Control__: `python create_dataset.py --config-name joystick_control_config`
<br>__RoNIN__:
<br> `python create_dataset.py --config-name ronin_train_config`
<br> `python create_dataset.py --config-name ronin_test_config`
<br>__VECtor__: `python create_dataset.py --config-name vector_config`
<br>__TotalCapture__:
<br> `python create_dataset.py --config-name total_capture_train_config`
<br> `python create_dataset.py --config-name total_capture_test_config`



## Usage
To train HiSS models for sequential prediction, use the `train.py` file. For each dataset, we provide a `<dataset_name>_hiss_config.yaml` file in the `conf/` directory, containing model parameters corresponding to the best-performing HiSS model for the respective dataset. To train the model, simply run

```
python train.py --config-name <dataset_name>_hiss_config
```

New datasets can be added by creating a corresponding `Task` object in line with tasks defined in `vt_state/tasks`, and creating a config file in `conf/data_env/<data_env_name>`.
