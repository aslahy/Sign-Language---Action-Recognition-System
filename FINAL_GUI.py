import cv2
import numpy as np
import pickle
import time
import mediapipe as mp
import tkinter as tk
from tkinter import Button, Label, Entry, messagebox
from PIL import ImageTk, Image
from tensorflow.keras.models import load_model
import speech_recognition as sr
from pydub import AudioSegment
import arabic_reshaper
from bidi.algorithm import get_display
import os

# Load models and labels
with open('models/model.p', 'rb') as f:
    model_dict = pickle.load(f)
letter_model = model_dict['model']  # Model for letter recognition
action_model = load_model("models/action.h5")  # Model for action recognition
print("Models loaded successfully.")

actions = ["hello", "goodbye", "please", "thankyou", "yes", "no"]  # Action labels

# Initialize MediaPipe components
mp_hands = mp.solutions.hands
mp_holistic = mp.solutions.holistic
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

class ASLApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ASL and Action Detection")
        self.root.geometry("800x700")

        # Initial UI state and model-related variables
        self.mode = None
        self.running = False
        self.cap = None
        self.recording = False
        self.recognizer = sr.Recognizer()

        # initial conditions for sign-to-text
        self.sequence = ""
        self.action_sequence = []
        self.last_added_time = 0
        self.debounce_time = 2
        self.last_detection_time = time.time() - 4
        self.sentence = []
        self.sequence_length = 60
        self.threshold = 0.5

        # Button Initialization
        self.init_main_menu()

    def init_main_menu(self):
        # Clear the current window
        self.clear_window()

        #make sure camera is off
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

        # Main Menu Buttons
        self.mode = None
        self.main_label = Label(self.root, text="Sign Language Recognition App", font=("Arial", 20))
        self.main_label.pack()

        self.sign_to_text_btn = Button(self.root, text="Sign to Text", command=self.init_sign_to_text, width=20, height=2)
        self.sign_to_text_btn.pack()
        self.text_to_sign_btn = Button(self.root, text="Text to Sign", command=self.init_text_to_sign, width=20, height=2)
        self.text_to_sign_btn.pack()
        self.exit_btn = Button(self.root, text="Exit", command=self.root.quit, width=20, height=2)
        self.exit_btn.pack()

    def clear_window(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def extract_keypoints(self, results):
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 3)
        return rh          

    def init_sign_to_text(self):
        self.clear_window()
        self.mode = "Letter"  # Default to letter recognition

        # Create GUI elements
        self.video_frame = Label(self.root)
        self.video_frame.pack()

        self.text_label = Label(self.root, text="", font=("Arial", 24), fg="black", height=2)
        self.text_label.pack()

        # Control Buttons
        Button(self.root, text="Start", command=self.start_detection, width=20, height=2).pack()
        Button(self.root, text="Stop", command=self.stop_detection, width=20, height=2).pack()
        Button(self.root, text="Clear", command=self.clear_text, width=20, height=2).pack()
        Button(self.root, text="Switch Mode", command=self.switch_mode, width=20, height=2).pack()
        Button(self.root, text="Back", command=self.init_main_menu, width=20, height=2).pack()

        self.sequence = ""
        self.sentence = []
        self.debounce_time = 2
        self.last_added_time = time.time()

    def init_text_to_sign(self):
        self.clear_window()
        self.mode = "Text"
        
        # Text to Sign Mode UI
        self.text_entry = Entry(self.root, font=("Arial", 18))
        self.text_entry.pack()
        
        Button(self.root, text="Type", command=self.process_text_input, width=20, height=2).pack()
        Button(self.root, text="Speak", command=self.start_speech_to_text, width=20, height=2).pack()
        Button(self.root, text="Clear", command=lambda: self.text_entry.delete(0, 'end'), width=20, height=2).pack()
        Button(self.root, text="Back", command=self.init_main_menu, width=20, height=2).pack()


    def process_arabic_text(self, text):
        """Process Arabic text for proper RTL display"""
        try:
            reshaped_text = arabic_reshaper.reshape(text)
            bidi_text = get_display(reshaped_text)
            return bidi_text
        except Exception as e:
            print(f"Error processing Arabic text: {e}")
            return text

    def start_speech_to_text(self):
        """Record and process speech input"""
        try:
            with sr.Microphone() as source:
                self.text_label.config(text="Start speaking")
                self.root.update()
                
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                
                # Record audio
                audio = self.recognizer.listen(source, timeout=5)
                
                self.text_label.config(text="Processing")
                self.root.update()
                
                # Recognize speech
                text = self.recognizer.recognize_google(audio, language='en-IN')
                
                # Process text for RTL display
                formatted_text = self.process_arabic_text(text)
                
                # Update UI
                self.text_entry.delete(0, tk.END)
                self.text_entry.insert(0, formatted_text)
                self.text_label.config(text="")
                
        except sr.UnknownValueError:
            self.text_label.config(text="Word not recognized")
        except sr.RequestError as e:
            self.text_label.config(text=f"Request Error : {str(e)}")
        except Exception as e:
            self.text_label.config(text=f"Exception : {str(e)}")

    def init_text_to_sign(self):
        self.clear_window()
        self.mode = "Text"
        
        # Configure RTL support for text entry
        self.text_entry = Entry(self.root, font=("Arial", 18), justify="right")
        self.text_entry.pack()
        
        # Add text label for status messages
        self.text_label = Label(self.root, text="", font=("Arial", 14))
        self.text_label.pack()
        
        Button(self.root, text="Type", command=self.process_text_input, width=20, height=2).pack()
        Button(self.root, text="Speak", command=self.start_speech_to_text, width=20, height=2).pack()
        Button(self.root, text="Clear", command=lambda: self.text_entry.delete(0, 'end'), width=20, height=2).pack()
        Button(self.root, text="Back", command=self.init_main_menu, width=20, height=2).pack()
        Button(self.root, text="Finish", command=self.process_text_input, width=20, height=2).pack()

    def process_text_input(self):
        """Process text input and find corresponding sign video"""
        text = self.text_entry.get().strip()
        if not text:
            self.text_label.config(text="Please enter the text")
            return
            
        # Process text for video lookup
        processed_text = text.replace(" ", "_").lower()[::-1]
        processed_text = arabic_reshaper.reshape(processed_text)
        print(processed_text)
        video_path = f'signs/{processed_text}.mp4'
        
        if os.path.exists(video_path):
            self.play_video(video_path)
        else:
            self.text_label.config(text="Word not in dictionary")

    # [Rest of the existing methods remain unchanged]

    def play_video(self, video_path):
        """Play sign language video with error handling"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("Could not open video file")
                
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                cv2.imshow("Sign", frame)
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
                    
            cap.release()
            cv2.destroyAllWindows()
            
        except Exception as e:
            self.text_label.config(text=f"Error in video processing: {str(e)}")

    def start_detection(self):
        if not self.running:
            self.cap = cv2.VideoCapture(0)
            self.running = True
            self.update_video()

    def stop_detection(self):
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def clear_text(self):
        self.sequence = ""
        self.sentence = []
        self.text_label.config(text="")

    def switch_mode(self):
        if self.mode == "Letter":
            self.mode = "Action"
            self.text_label.config(text="Switched to Action Detection")
        else:
            self.mode = "Letter"
            self.text_label.config(text="Switched to Letter Detection")

    def update_video(self):
        if not self.running:
            return
        ret, frame = self.cap.read()
        if not ret:
            return

        if self.mode == "Letter":
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    x_ = [lm.x for lm in hand_landmarks.landmark]
                    y_ = [lm.y for lm in hand_landmarks.landmark]
                    data_aux = [(lm.x - min(x_), lm.y - min(y_)) for lm in hand_landmarks.landmark]
                    prediction = letter_model.predict([np.asarray(data_aux).flatten()])
                    predicted_character = prediction[0]

                    current_time = time.time()
                    if current_time - self.last_added_time > self.debounce_time:
                        self.sequence += predicted_character
                        self.text_label.config(text=self.sequence)
                        self.last_added_time = current_time

                    # Draw landmarks
                    for lm in hand_landmarks.landmark:
                        cx, cy = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                        cv2.circle(frame, (cx, cy), 5, (0, 0, 0), -1)


        elif self.mode == "Action":
            current_time = time.time()

            # Check cooldown period
            if current_time - self.last_detection_time >= 3:  # 3 seconds cooldown
                results = holistic.process(frame)
                cv2.circle(frame, (30, 450), 20, (0, 255, 0), -1)  # Green indicator during detection

                if results:
                    keypoints = self.extract_keypoints(results)
                    
                    # Append keypoints to the sequence for action prediction
                    self.action_sequence.append(keypoints)
                    
                    # Maintain length at 30 frames by popping old data
                    if len(self.action_sequence) > 30:
                        self.action_sequence.pop(0)

                    # Only predict if we have exactly 30 frames
                    if len(self.action_sequence) == 30:
                        # Convert to 3D array and predict
                        sampled_sequence = self.action_sequence  
                        input_data = np.expand_dims(sampled_sequence, axis=0)
                        res = action_model.predict(input_data)[0]

                        # Add action if above threshold
                        if res[np.argmax(res)] > self.threshold:
                            action = actions[np.argmax(res)]
                            if not self.sentence or action != self.sentence[-1]:
                                self.sentence.append(action)
                            if len(self.sentence) > 5:
                                self.sentence = self.sentence[-5:]

                        # Update last detection time and display
                        self.last_detection_time = time.time()
                        self.text_label.config(text=" ".join(self.sentence))

                    # Draw landmarks
                    mp.solutions.drawing_utils.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    mp.solutions.drawing_utils.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            else:
                cv2.circle(frame, (30, 450), 20, (0, 0, 255), -1)  # Red indicator during cooldown
                self.action_sequence = []


        # Update GUI
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img_pil)
        self.video_frame.imgtk = imgtk
        self.video_frame.config(image=imgtk)

        self.root.after(10, self.update_video)

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    # Configure root window for RTL support
    root.tk.call('encoding', 'system', 'utf-8')
    app = ASLApp(root)
    root.mainloop()
