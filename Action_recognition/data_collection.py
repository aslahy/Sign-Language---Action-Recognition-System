import cv2
import os
import time

# Define actions and parameters
actions = ["hello", "goodbye", "please", "thankyou", "yes", "no"]
num_videos = 30
frames_per_video = 30
cooldown_time = 2  # seconds
video_size = (640, 480)  # Resolution for the videos
fps = 30  # Frames per second
output_dir = "actions"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Create directories for each action
for action in actions:
    action_dir = os.path.join(output_dir, action)
    os.makedirs(action_dir, exist_ok=True)

# Open the camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, video_size[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, video_size[1])
cap.set(cv2.CAP_PROP_FPS, fps)

print("Camera is on. Press 'q' to start recording each action.")

for action in actions:
    print(f"Get ready to record action: {action}")

    # Wait until 'q' is pressed to start recording
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        # Display red circle while waiting
        cv2.circle(frame, (video_size[0] - 50, video_size[1] - 50), 20, (0, 0, 255), -1)
        cv2.imshow("Video Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Starting recording...")
            break

    for video_idx in range(50,num_videos+50):
        print(f"Recording video {video_idx + 1}/{num_videos} for action '{action}'")
        video_path = os.path.join(output_dir, action, f"{video_idx}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
        out = cv2.VideoWriter(video_path, fourcc, fps, video_size)

        # Record video
        for frame_idx in range(frames_per_video):
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame during recording. Exiting...")
                break

            # Display green circle while recording
            cv2.circle(frame, (video_size[0] - 50, video_size[1] - 50), 20, (0, 255, 0), -1)
            cv2.imshow("Video Feed", frame)
            out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Recording interrupted by user. Exiting...")
                break

        out.release()
        print(f"Finished video {video_idx + 1}/{num_videos} for action '{action}'")

        # Cooldown period
        print("Cooldown period...")
        cooldown_end = time.time() + cooldown_time
        while time.time() < cooldown_end:
            ret, frame = cap.read()
            if not ret:
                break
            # Display red circle during cooldown
            cv2.circle(frame, (video_size[0] - 50, video_size[1] - 50), 20, (0, 0, 255), -1)
            cv2.imshow("Video Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Cooldown interrupted by user.")
                break

    print(f"Completed recording for action: {action}")

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("All recordings complete.")
