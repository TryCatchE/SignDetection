import os
import pickle

import mediapipe as mp
import cv2

hands = mp.solutions.hands
drawingUtls = mp.solutions.drawing_utils
stylesUtls = mp.solutions.drawing_styles

handsObj = hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.3)

imgsDir = './doubleHands'

data = []
labels = []
for imgDir in os.listdir(imgsDir):
    for imgPath in os.listdir(os.path.join(imgsDir, imgDir)):
        img = cv2.imread(os.path.join(imgsDir, imgDir, imgPath))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = handsObj.process(img_rgb)
        if results.multi_hand_landmarks:
            all_hands_data = []
            for handCordinates in results.multi_hand_landmarks:
                hand_data = []
                xList = [landmark.x for landmark in handCordinates.landmark]
                yList = [landmark.y for landmark in handCordinates.landmark]

                for i in range(len(handCordinates.landmark)):
                    x = handCordinates.landmark[i].x
                    y = handCordinates.landmark[i].y
                    hand_data.append(x - min(xList))
                    hand_data.append(y - min(yList))

                all_hands_data.extend(hand_data)

            # Ensure 84 features (for two hands)
            if len(all_hands_data) < 84:
                all_hands_data.extend([0] * (84 - len(all_hands_data)))

            data.append(all_hands_data)
            labels.append(imgDir)

f = open('data2.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
