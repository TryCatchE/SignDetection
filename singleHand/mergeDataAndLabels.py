import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

# Initialize mediapipe Hands module
hands = mp.solutions.hands
drawingUtls = mp.solutions.drawing_utils
stylesUtls = mp.solutions.drawing_styles

# Create a hands detection object
handsObj = hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Directory containing images
imgsDir = './images'

# Lists to store data and corresponding labels
data = []
labels = []

# Iterate through each directory (each representing a label)
for imgDir in os.listdir(imgsDir):
    # Iterate through each image in the directory
    for imgPath in os.listdir(os.path.join(imgsDir, imgDir)):
        edgesOfHand = []  # List to store relative coordinates of hand landmarks

        xList = []  # List to store x-coordinates of hand landmarks
        yList = []  # List to store y-coordinates of hand landmarks

        # Read the image
        img = cv2.imread(os.path.join(imgsDir, imgDir, imgPath))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image to detect hands
        results = handsObj.process(img_rgb)

        # Check if hands are detected in the image
        if results.multi_hand_landmarks:
            # Iterate through each detected hand
            for handCordinates in results.multi_hand_landmarks:
                # Iterate through each landmark of the hand
                for i in range(len(handCordinates.landmark)):
                    # Get x and y coordinates of the landmark
                    x = handCordinates.landmark[i].x
                    y = handCordinates.landmark[i].y

                    # Append coordinates to respective lists
                    xList.append(x)
                    yList.append(y)

                # Calculate relative coordinates by subtracting minimum x and y coordinates
                for i in range(len(handCordinates.landmark)):
                    x = handCordinates.landmark[i].x
                    y = handCordinates.landmark[i].y
                    edgesOfHand.append(x - min(xList))
                    edgesOfHand.append(y - min(yList))

            # Append relative coordinates to data list and label to labels list
            data.append(edgesOfHand)
            labels.append(imgDir)

# Save the data and labels to a pickle file
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
