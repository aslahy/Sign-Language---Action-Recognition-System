import os
import pickle
import mediapipe as mp
import cv2
from tqdm import tqdm  # Import tqdm for progress bar visualization

# Initialize MediaPipe Hands model for hand landmark detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './master_dataset'  # Directory containing labeled sign images

# Initialize lists for storing data, labels, and skipped image IDs
data = []  # List to store processed hand landmarks
labels = []  # List to store corresponding labels (class names)
skipped_images = []  # List to store IDs of skipped images (sign name and image number)

# Get all directories in the dataset folder
directories = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]

# Iterate through each directory (class label)
for dir_ in tqdm(directories, desc="Processing Directories"):
    dir_path = os.path.join(DATA_DIR, dir_)

    # Get all images in the current directory
    images = os.listdir(dir_path)
    
    # Iterate through each image in the directory
    for img_num, img_path in enumerate(tqdm(images, desc=f"Processing Images in {dir_}", leave=False), start=1):
        data_aux = []  # Temporary list for normalized landmark data
        x_ = []  # List to collect x-coordinates for normalization
        y_ = []  # List to collect y-coordinates for normalization

        # Load and preprocess the image
        img = cv2.imread(os.path.join(dir_path, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert image to RGB format

        # Process the image to detect hand landmarks
        results = hands.process(img_rgb)
        
        # Check if landmarks are detected
        if results.multi_hand_landmarks:
            valid_hand = True  # Assume the hand is valid

            for hand_landmarks in results.multi_hand_landmarks:
                # Collect x and y coordinates for all landmarks
                for lm in hand_landmarks.landmark:
                    if lm.x == 0 or lm.y == 0:  # Check for missing landmarks
                        valid_hand = False
                        break
                    x_.append(lm.x)
                    y_.append(lm.y)

                if not valid_hand:
                    break

                # Normalize coordinates by subtracting the minimum values
                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_))  # Normalize x
                    data_aux.append(lm.y - min(y_))  # Normalize y

            if valid_hand:
                # Append processed data and label to their respective lists
                data.append(data_aux)
                labels.append(dir_)
            else:
                # Log skipped image with its ID (sign name and image number)
                skipped_images.append(f"{dir_}/{img_num}")
        else:
            # Log skipped image if no landmarks are detected
            skipped_images.append(f"{dir_}/{img_num}")

# Save the processed data and labels to a pickle file for later use
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

# Save the list of skipped images to a text file
with open('skipped_images.txt', 'w') as f:
    for img_id in skipped_images:
        f.write(f"{img_id}\n")

print(f"Data processing completed. {len(skipped_images)} images were skipped. Check 'skipped_images.txt' for details.")
