import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import threading

# Load the saved model
modelDir = pickle.load(open('./model.p', 'rb'))
model = modelDir['model']

# Initialize Text-to-Speech engine
engine = pyttsx3.init()
engine_busy = False  # Flag to track if the engine is currently speaking

cap = cv2.VideoCapture(0)

hands = mp.solutions.hands
drawUtls = mp.solutions.drawing_utils
styleUtls = mp.solutions.drawing_styles

handsObj = hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels = {0: 'SANDY', 1: 'TI', 2: 'KANEIS'}

def speak_label(label):
    global engine_busy
    if not engine_busy:
        engine_busy = True
        engine.say(label)
        engine.runAndWait()
        engine_busy = False

while True:
    handEdges = []
    xList = []
    yList = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    vFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = handsObj.process(vFrame)
    if results.multi_hand_landmarks:
        for handCordinates in results.multi_hand_landmarks:
            drawUtls.draw_landmarks(
                frame,  # image to draw
                handCordinates,  # model output
                hands.HAND_CONNECTIONS,  # hand connections
                styleUtls.get_default_hand_landmarks_style(),
                styleUtls.get_default_hand_connections_style())

        for handCordinates in results.multi_hand_landmarks:
            for i in range(len(handCordinates.landmark)):
                x = handCordinates.landmark[i].x
                y = handCordinates.landmark[i].y

                xList.append(x)
                yList.append(y)

            for i in range(len(handCordinates.landmark)):
                x = handCordinates.landmark[i].x
                y = handCordinates.landmark[i].y
                handEdges.append(x - min(xList))
                handEdges.append(y - min(yList))

        x1 = int(min(xList) * W) - 10
        y1 = int(min(yList) * H) - 10

        x2 = int(max(xList) * W) - 10
        y2 = int(max(yList) * H) - 10

        prediction = model.predict([np.asarray(handEdges)])

        predLabel = labels[int(prediction[0])]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predLabel, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

        # Speak out the prediction label in a separate thread
        threading.Thread(target=speak_label, args=(predLabel,)).start()

    cv2.imshow('frame', frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
