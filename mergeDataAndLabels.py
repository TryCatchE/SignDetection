import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt


hands = mp.solutions.hands
drawingUtls = mp.solutions.drawing_utils
stylesUtls = mp.solutions.drawing_styles

handsObj = hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

imgsDir = './images'

data = []
labels = []
for imgDir in os.listdir(imgsDir):
    for imgPath in os.listdir(os.path.join(imgsDir, imgDir)):
        edgesOfHand = []

        xList = []
        yList = []

        img = cv2.imread(os.path.join(imgsDir, imgDir, imgPath))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = handsObj.process(img_rgb)
        if results.multi_hand_landmarks:
            for handCordinates in results.multi_hand_landmarks:
                for i in range(len(handCordinates.landmark)):
                    
                    x = handCordinates.landmark[i].x
                    y = handCordinates.landmark[i].y

                    xList.append(x)
                    yList.append(y)
    # roduce the same set of relative coordinates even if it is shifted within the image
    # preprocessing step to make the data more invariant to translation
                for i in range(len(handCordinates.landmark)):
                    x = handCordinates.landmark[i].x
                    y = handCordinates.landmark[i].y
                    edgesOfHand.append(x - min(xList))
                    edgesOfHand.append(y - min(yList))

            data.append(edgesOfHand)
            labels.append(imgDir)

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
