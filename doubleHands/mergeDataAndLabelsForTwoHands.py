import os
import pickle
import mediapipe as mp
import cv2

# Initialize the mediapipe Hands module
hands = mp.solutions.hands
drawingUtls = mp.solutions.drawing_utils
stylesUtls = mp.solutions.drawing_styles

# Initialize the Hands object with desired parameters
handsObj = hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3)

# Directory containing the images
imgsDir = './doubleHands'

# Lists to store hand data and corresponding labels
data = []
labels = []

# Iterate through each directory in the main directory
for imgDir in os.listdir(imgsDir):
    # Iterate through each image in the subdirectories
    for imgPath in os.listdir(os.path.join(imgsDir, imgDir)):
        # Read the image using OpenCV
        img = cv2.imread(os.path.join(imgsDir, imgDir, imgPath))
        # Convert image to RGB format (required by mediapipe)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image using mediapipe Hands module
        results = handsObj.process(img_rgb)
        if results.multi_hand_landmarks:
            # List to store all hand landmark data
            all_hands_data = []
            # Iterate through each detected hand in the image
            for handCordinates in results.multi_hand_landmarks:
                # List to store landmark coordinates for a single hand
                hand_data = []
                # Extract x and y coordinates of each landmark and normalize them
                xList = [landmark.x for landmark in handCordinates.landmark]
                yList = [landmark.y for landmark in handCordinates.landmark]

                for i in range(len(handCordinates.landmark)):
                    x = handCordinates.landmark[i].x
                    y = handCordinates.landmark[i].y
                    # Normalize coordinates relative to the bounding box
                    hand_data.append(x - min(xList))
                    hand_data.append(y - min(yList))

                # Append hand data for a single hand to the list
                all_hands_data.extend(hand_data)

            # Ensure each hand has 84 features (landmark coordinates)
            if len(all_hands_data) < 84:
                all_hands_data.extend([0] * (84 - len(all_hands_data)))

            # Append hand data and corresponding label to the lists
            data.append(all_hands_data)
            labels.append(imgDir)

# Save the data and labels to a pickle file
f = open('data2.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
