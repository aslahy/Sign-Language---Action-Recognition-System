import os
import shutil

# Paths for master_dataset and the new samples folder
master_dataset_path = './master_dataset'
samples_folder_path = './samples'

# Create the samples folder if it doesn't exist
os.makedirs(samples_folder_path, exist_ok=True)

# Iterate through each sign folder in master_dataset
for sign_folder in os.listdir(master_dataset_path):
    sign_folder_path = os.path.join(master_dataset_path, sign_folder)
    if os.path.isdir(sign_folder_path):
        # Get a list of all files in the sign folder
        files = os.listdir(sign_folder_path)
        if files:
            # Pick the first image file as the sample
            sample_image_path = os.path.join(sign_folder_path, files[0])
            if os.path.isfile(sample_image_path):
                target_sample_path = os.path.join(samples_folder_path, f"{sign_folder}.png")
                shutil.copy(sample_image_path, target_sample_path)

print("Samples have been successfully created in the samples folder.")
