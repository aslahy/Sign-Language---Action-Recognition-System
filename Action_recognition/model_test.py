import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
import time

# Load the pre-trained model
model = load_model("action.h5")
print("Model loaded successfully.")

# Define actions and colors
actions = ["hello", "goodbye", "please", "thankyou", "yes", "no"]
colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]
sequence_length = 30  
threshold = 0.5

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize Mediapipe Holistic model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Function to detect and draw landmarks
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# Keypoint extraction without face and pose landmarks
def extract_keypoints(results):
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
    return rh

# Start processing video feed
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    sequence = []
    sentence = []
    predictions = []
    data_collecting = False
    last_data_collection_time = time.time() - 3  # Allow immediate data collection initially

    while cap.isOpened():
        # Read feed
        ret, frame = cap.read()
        if not ret:
            break

        # Determine if we can collect data
        current_time = time.time()
        if current_time - last_data_collection_time > 3:
            data_collecting = True
            indicator_color = (0, 255, 0)  # Green when collecting data
        else:
            data_collecting = False
            indicator_color = (0, 0, 255)  # Red when not collecting data

        # Make detections
        image, results = mediapipe_detection(frame, holistic)

        # Draw landmarks
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Collect data if allowed
        if data_collecting:
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)

            # If we have collected 60 frames, take every second frame to get 30 frames for prediction
            if len(sequence) == 30:
                sampled_sequence = sequence
                print(len(sampled_sequence))  # Should print 30

                # Prediction logic
                res = model.predict(np.expand_dims(sampled_sequence, axis=0))[0]
                predictions.append(np.argmax(res))

                # Check if predictions are consistent
                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:
                        if len(sentence) > 0:
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                # Ensure sentence length does not exceed 5 words
                if len(sentence) > 5:
                    sentence = sentence[-5:]

                # Reset sequence and predictions, stop collecting data
                sequence = []
                predictions = []
                last_data_collection_time = time.time()  # Record the time of this data collection
                data_collecting = False  # Enter cooldown period

        # Display the sentence at the top
        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the indicator light in the bottom-left corner
        cv2.circle(image, (30, 450), 20, indicator_color, -1)

        # Show the frame
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
