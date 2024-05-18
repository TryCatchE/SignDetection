# Hand Gesture Recognition System

## Description
This project is a real-time hand gesture recognition system using Python and various libraries such as OpenCV, MediaPipe, and scikit-learn.

## Features
- Real-time hand gesture recognition
- Support for single and double hand gestures
- Recognition of predefined gestures for letters, words, and commands
- Dynamic display of recognized gestures and actions

## Dependencies
Make sure you have the following dependencies installed:
- Python 3.x
- OpenCV
- MediaPipe
- scikit-learn
- pyttsx3
- numpy

## Usage
1. Clone the repository:
   ```
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```
   cd hand-gesture-recognition
   ```
3. Run the main script:
   ```
   python runModels.py
   ```
4. Use hand gestures in front of your webcam to interact with the system.

## Models
The trained machine learning models are stored in the models directory. Ensure that the models are available in this directory before running the system.

## Collecting Training Data
To collect training data for custom hand gestures, follow these steps:
1. Run the script `openCV.py` or `openCvForTwoHands.py`.
2. Press 'Q' to start capturing images for each hand gesture type.
3. Collect a sufficient number of images for each gesture type (default: 100 images per type).
4. Images will be saved in the images directory.

## Prepare Data
Before training the model, run the `mergeDataAndLabels.py` or `mergeDataAndLabelsForTwoHands.py`. These scripts are responsible for merging the data with some labels.

## Training the Models
To train the model using the collected data, run the script `trainModel.py` or `trainModelForTwoHands.py`. The trained model will be saved as a pickle file in the models directory.

## Evaluation
The performance of the trained model can be evaluated using the same script as the training. This script calculates accuracy, generates a confusion matrix, plots a learning curve, and provides a classification report.

## Acknowledgments
This project utilizes the following libraries and frameworks:
- [OpenCV](https://opencv.org/)
- [MediaPipe](https://mediapipe.dev/)
- [scikit-learn](https://scikit-learn.org/)
- [pyttsx3](https://pypi.org/project/pyttsx3/)
- [numpy](https://numpy.org/)
