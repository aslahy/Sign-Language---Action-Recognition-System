# Sign detection using Mediapipe

Sign language detector with Python, OpenCV and Mediapipe !

Inspired from **Computer vision engineer**'s YouTube video 
[**Sign language detection with Python and Scikit Learn | Landmark detection | Computer vision tutorial**](https://www.youtube.com/watch?v=MJCSjXepaAM)


# Folder Organization

This folder contains 5 folders and 12 files.

The folders are:-
asl_dataset
asy_dataset
behy_dataset
master_dataset
samples

The files are:-
image_tester.ipynb
collect_imgs.py
image_dataset_extractor.py
sample_generator.py
create_training_data.py
data.pickle
skipped_images.txt
skip_view.py
train_cassifier.py
model.p
inference_classifier.py



## Dataset folders

The asl_dataset is a kaggle dataset containing 70 images for each letter.
The other datasets ( asy, behy) are datasets containing 250 images for each letter each by different signers for variety in training data.
The master_dataset is a dataset which contains all the 3 above datasets together and is directly used to create training data.

## Samples folder

The samples folder contains 1 image of each sign for reference.

## image_tester.ipynb

This file is used to test if some image in the dataset is providing landmarks correctly or not to determine if the dataset is usable. You can try giving different file paths to test different images for landmarks. Some datasets available online will be of poor quality and landmark extraction will not be possible for these datasets. 
Also it can test your webcam to see if the webcam output can be used to make datasets depending on if landmarks are visible.

## collect_imgs.py

Collects 250 images continuously for each sign using the computer webcam. Stores it in required folder. A sample image will be displayed on the side for reference to sign.

## image_dataset_extractor.py

This file extracts images from selected folder and adds it to the master_dataset for easy use. This can help us in easily integrating online available datasets and locally made datasets.

## sample_generator.py

This file makes the folder samples which has 1 image of each sign, which can be used for easy identification. When the collect_imgs.py file is run, the corresponding image from samples folder for the letter will be shown, so that we can easily create local datasets.

## create_training_dataset.py

This file reads the master_dataset folder and extract landmarks for each images, excluding images where all landmarks are present. Writes this data into the data.pickle file for later use. Also makes a text file skipped_images.txt to keep track of images with incomplete landmarks. This information will be useful for some analysis ( skewed input because some letter may have sore images skipped, reason for image skipping).
This process may take some time so there is a progress bar implemented using tqdm.

## skip_view.py

Plots skipped images corresponding to it's label letter. Helps understand skewing in data.

## train_cassifier.py

This file trains the classifier using the data.pickle file and generates the model model.p

## inference_cassifier.py

This file uses the model.p model to guess the sign shown in real-time.
