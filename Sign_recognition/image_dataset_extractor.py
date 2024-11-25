import os
import shutil

# Paths for source and destination directories
source_dataset_path = './behy_dataset'
master_dataset_path = './master_dataset'

# Create the master_dataset folder if it doesn't exist
os.makedirs(master_dataset_path, exist_ok=True)

# Process each folder in asl_dataset
for sign_folder in os.listdir(source_dataset_path):
    sign_folder_path = os.path.join(source_dataset_path, sign_folder)
    if os.path.isdir(sign_folder_path):
        # Create corresponding folder in master_dataset
        target_folder_path = os.path.join(master_dataset_path, sign_folder)
        os.makedirs(target_folder_path, exist_ok=True)
        
        # Determine the next available number in the target folder
        existing_files = os.listdir(target_folder_path)
        existing_numbers = [
            int(os.path.splitext(f)[0]) for f in existing_files if f.split(".")[0].isdigit()
        ]
        next_number = max(existing_numbers, default=-1) + 1
        
        # Copy and rename images
        for image_file in os.listdir(sign_folder_path):
            source_image_path = os.path.join(sign_folder_path, image_file)
            if os.path.isfile(source_image_path):
                while True:
                    target_image_path = os.path.join(target_folder_path, f"{next_number}.png")
                    if not os.path.exists(target_image_path):  # Ensure unique naming
                        shutil.copy(source_image_path, target_image_path)
                        next_number += 1
                        break

print("Images have been successfully organized in the master_dataset folder.")
