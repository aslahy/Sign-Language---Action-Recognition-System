import os
import cv2
import sys

DATA_DIR = './behy_dataset'
SAMPLES_DIR = './samples'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

classes = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
           'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
dataset_size = 250

# Attempt to access the camera, if this doesn't work try 1,2,...
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not accessible. Check if the camera is connected and try again.")
else:
    for j in classes:
        # Create directory for each class if it does not exist
        class_dir = os.path.join(DATA_DIR, j)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        print('Collecting data for class {}'.format(j))

        # Load the sample image for the current class
        sample_path = os.path.join(SAMPLES_DIR, f"{j}.png")
        if os.path.exists(sample_path):
            sample_image = cv2.imread(sample_path)
            cv2.imshow('Sample Image', sample_image)
        else:
            print(f"No sample image found for class {j}.")
            cv2.destroyWindow('Sample Image')

        # Display "Ready? Press Q" prompt to the user
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture image. Please check the camera.")
                break

            cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3,
                        cv2.LINE_AA)
            cv2.imshow('Camera Feed', frame)

            if cv2.waitKey(25) == ord('q'):
                break
            if cv2.waitKey(25) == ord('e'):
                exit(0)

        # Capture specified number of images for the current class
        counter = 0
        while counter < dataset_size:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture image.")
                break

            cv2.imshow('Camera Feed', frame)
            cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)
            counter += 1
            cv2.waitKey(25)

        cv2.destroyWindow('Sample Image')  # Close the sample window after each class

cap.release()
cv2.destroyAllWindows()
