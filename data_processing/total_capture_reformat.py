import os
import shutil
import tarfile
import sys

sys.path.append("./")

from hiss.utils import DATA_DIR

def decompress_tar_files(folder_path):
    for item in os.listdir(folder_path):
        # Check if the file is a .tar or .tar.gz file
        if item.endswith(".tar") or item.endswith(".tar.gz"):
            tar_path = os.path.join(folder_path, item)
            # Remove the file extension for the folder name (.tar or .tar.gz)
            if item.endswith(".tar.gz"):
                extract_dir = os.path.join(folder_path, item[:-7])  # Remove '.tar.gz'
            else:
                extract_dir = os.path.join(folder_path, item[:-4])  # Remove '.tar'

            # Create a directory for the extracted files if it doesn't exist
            os.makedirs(extract_dir, exist_ok=True)

            # Extract the tar file
            with tarfile.open(tar_path) as tar:
                tar.extractall(path=extract_dir)
                print(f"Extracted '{tar_path}' to '{extract_dir}'")



def rename_subfolders_to_lowercase(folder_path):
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for name in dirs:
            if name[0].isupper():  # Check if the folder name is in uppercase
                original_path = os.path.join(root, name)
                new_path = os.path.join(root, name.lower())
                os.rename(original_path, new_path)  # Rename the folder to lowercase
                print(f"Renamed '{original_path}' to '{new_path}'")


if __name__ == "__main__":
    # Base directory where the original files are located
    base_dir = os.path.join(DATA_DIR, "total_capture")
    decompress_tar_files(base_dir)
    rename_subfolders_to_lowercase(base_dir)
    # Training and testing directories
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")

    # Subjects and their sequences for training and testing
    train_subjects = ["s1", "s2", "s3"]
    train_sequences = [
        "rom1",
        "rom2",
        "rom3",
        "walking1",
        "walking3",
        "freestyle1",
        "freestyle2",
        "acting1",
        "acting2",
    ]

    test_subjects = ["s1", "s2", "s3", "s4", "s5"]
    test_sequences = ["walking2", "freestyle3", "acting3"]

    for subject_id in set(train_subjects + test_subjects):
        imu_base_dir = os.path.join(base_dir, f"{subject_id}_imu", subject_id)
        vicon_base_dir = os.path.join(
            base_dir, f"{subject_id}_vicon_pos_ori", subject_id
        )

        for seq in set(train_sequences + test_sequences):
            new_dir_created = False

            # Determine if the current iteration is for training or testing
            if subject_id in train_subjects and seq in train_sequences:
                target_dir = train_dir
            elif subject_id in test_subjects and seq in test_sequences:
                target_dir = test_dir
            else:
                continue  # Skip sequences and subjects not part of the current partition

            # Check and move IMU files
            imu_files = [
                f"{subject_id}_{seq}_Xsens.sensors",
                f"{subject_id}_{seq}_calib_imu_ref.txt",
            ]
            for imu_file in imu_files:
                original_path = os.path.join(imu_base_dir, imu_file)
                if os.path.exists(original_path):
                    if not new_dir_created:
                        new_dir_path = os.path.join(target_dir, f"{subject_id}_{seq}")
                        os.makedirs(new_dir_path, exist_ok=True)
                        new_dir_created = True

                    new_file_name = (
                        "imu.txt"
                        if "Xsens.sensors" in imu_file
                        else imu_file.replace(f"{subject_id}_{seq}_", "")
                    )
                    new_path = os.path.join(new_dir_path, new_file_name)
                    shutil.move(original_path, new_path)

            # Check and move Vicon files
            vicon_files = ["gt_skel_gbl_ori.txt", "gt_skel_gbl_pos.txt"]
            for vicon_file in vicon_files:
                original_path = os.path.join(vicon_base_dir, seq, vicon_file)
                if os.path.exists(original_path):
                    if not new_dir_created:
                        new_dir_path = os.path.join(target_dir, f"{subject_id}_{seq}")
                        os.makedirs(new_dir_path, exist_ok=True)
                        new_dir_created = True

                    new_path = os.path.join(new_dir_path, vicon_file)
                    shutil.move(original_path, new_path)

    print("Data split into training and testing sets complete.")
