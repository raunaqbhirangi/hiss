import os
import shutil

import sys

sys.path.append("./")

from hiss.utils import DATA_DIR

if __name__ == "__main__":
    # Replace 'your_folder_path_here' with the path to your folder containing the files
    folder_path = os.path.join(DATA_DIR, "vector")

    for filename in os.listdir(folder_path):
        # Check if the file is of interest (either .imu.txt or .gt.txt)
        if ".synced.imu.txt" in filename or ".synced.gt.txt" in filename:
            base_name, file_type = filename.split(".synced")
            subdir_name = os.path.join(folder_path, base_name)
            new_filename = "imu.txt" if "imu" in file_type else "gt.txt"

            # Create the subdirectory if it doesn't exist
            os.makedirs(subdir_name, exist_ok=True)

            # Construct the full paths for source and destination
            src_path = os.path.join(folder_path, filename)
            dest_path = os.path.join(subdir_name, new_filename)

            # Move and rename the file
            shutil.move(src_path, dest_path)

    print("Files have been reorganized and renamed.")
