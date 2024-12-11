
# Action recognition using Mediapipe

  

Action detector with Python, OpenCV and Mediapipe !

  

Inspired from **Nicholas Renotte**'s YouTube video

[**Sign Language Detection using ACTION RECOGNITION with Python | LSTM Deep Learning Model**](https://www.youtube.com/watch?v=doDUihpj6ro&t=754s)

  
  

# Folder Organization

  

This folder contains 1 folder and 5 files.

The folder actions contain all the data collected using the data_collection.py file

The files are:-

data_collection.py
action_recognition.ipynb
model_processing.ipynb
action.h5
model_test.py

  
  
  

## data_collection.py

  

Collects 30 frame long videos at 30 frames per second for 6 different actions.

  
## action_recognition.py

Extracts hand landmarks from each frame in each video and stores as .npy files.


## model_processing.ipynb
Trains the model on the data stored in the .npy file using a deep learning based LSTM model.
  

## model_test.py


Tests the action.h5 model in real-time.
