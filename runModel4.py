import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import threading
import string
import time

# Load the saved models
model_single_hand_dir = pickle.load(open('./model.p', 'rb'))  # Adjust path as needed
model_double_hands_dir = pickle.load(open('./model2.p', 'rb'))  # Adjust path as needed

model_single_hand = model_single_hand_dir['model']
model_double_hands = model_double_hands_dir['model']

# Initialize Text-to-Speech engine
engine = pyttsx3.init()
engine_busy = False  # Flag to track if the engine is currently speaking

cap = cv2.VideoCapture(0)

hands = mp.solutions.hands
drawUtls = mp.solutions.drawing_utils
styleUtls = mp.solutions.drawing_styles

handsObj = hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3)

labels_single_hand = {i: letter for i, letter in enumerate(string.ascii_uppercase)}  # Adjust labels as needed
labels_double_hands = {0: 'open text', 1: 'clear', 2: 'speak'}  # Adjust labels as needed

prev_label = None  # Variable to store the previous prediction label
label_start_time = None  # Time when the current label was first detected
stored_predictions = []  # Array to store predictions
display_black_box = False  # Flag to control the display of the black box
black_box_content = ""  # Content to display in the black box
last_stored_prediction = None  # Variable to store the last stored prediction
last_stored_time = None  # Time when the last prediction was stored
store_interval = 5  # Interval in seconds to store new predictions

# New variables for gesture tracking
current_gesture = None
gesture_start_time = None
PREDICTION_INTERVAL = 1.5  # Time in seconds to wait before making a prediction

import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import threading
import string
import time

# Load the saved models
model_single_hand_dir = pickle.load(open('./model.p', 'rb'))  # Adjust path as needed
model_double_hands_dir = pickle.load(open('./model2.p', 'rb'))  # Adjust path as needed

model_single_hand = model_single_hand_dir['model']
model_double_hands = model_double_hands_dir['model']

# Initialize Text-to-Speech engine
engine = pyttsx3.init()
engine_busy = False  # Flag to track if the engine is currently speaking

cap = cv2.VideoCapture(0)

hands = mp.solutions.hands
drawUtls = mp.solutions.drawing_utils
styleUtls = mp.solutions.drawing_styles

handsObj = hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3)

labels_single_hand = {i: letter for i, letter in enumerate(string.ascii_uppercase)}  # Adjust labels as needed
labels_double_hands = {0: 'open text', 1: 'clear', 2: 'speak'}  # Adjust labels as needed

# New variables for gesture tracking
current_gesture = None
gesture_start_time = None
PREDICTION_INTERVAL = 1.5  # Time in seconds to wait before making a prediction

def speak_labels(labels):
    global engine_busy
    if not engine_busy:
        engine_busy = True

        # Initialize an empty string to accumulate letters into words
        word = ''
        # Initialize a list to store complete words
        words = []

        for label in labels:
            if label == '_':  # When an underscore is encountered, start a new word
                if word:  # Add the accumulated word to the list if it's not empty
                    words.append(word)
                    word = ''  # Reset the word accumulator
            else:
                word += label  # Accumulate letters into a word

        # Add the last accumulated word if it's not empty
        if word:
            words.append(word)

        # Speak each word
        for word in words:
            engine.say(word)
            engine.runAndWait()
            print(word)

        engine_busy = False

while True:
    handEdges = []
    xList = []
    yList = []

    ret, frame = cap.read()

    if not ret:
        break

    H, W, _ = frame.shape

    vFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = handsObj.process(vFrame)
    num_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0

    if num_hands > 0:
        for handCordinates in results.multi_hand_landmarks:
            drawUtls.draw_landmarks(
                frame,
                handCordinates,
                hands.HAND_CONNECTIONS,
                styleUtls.get_default_hand_landmarks_style(),
                styleUtls.get_default_hand_connections_style())

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

        if num_hands == 1:
            model = model_single_hand
            labels = labels_single_hand
            expected_num_features = 42  # Adjust as per your single hand model
        else:  # Assuming 2 hands
            model = model_double_hands
            labels = labels_double_hands
            expected_num_features = 84  # Adjust as per your double hands model

        if len(handEdges) > expected_num_features:
            handEdges = handEdges[:expected_num_features]
        elif len(handEdges) < expected_num_features:
            handEdges.extend([0] * (expected_num_features - len(handEdges)))

        input_features = np.asarray(handEdges).reshape(1, -1)
        prediction = model.predict(input_features)
        predLabel = labels[int(prediction[0])]

        x1, y1 = int(min(xList) * W), int(min(yList) * H)
        x2, y2 = int(max(xList) * W), int(max(yList) * H)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, predLabel, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Gesture duration checking
        if predLabel != current_gesture:
            current_gesture = predLabel
            gesture_start_time = time.time()
        elif time.time() - gesture_start_time >= PREDICTION_INTERVAL:
            # Now act on the gesture after 1.5 seconds
            if num_hands == 1 and predLabel != '_':
                stored_predictions.append(predLabel)
                black_box_content = ' '.join(stored_predictions)

            if num_hands == 2:
                if int(prediction[0]) == 0:
                    display_black_box = True
                elif int(prediction[0]) == 1:
                    display_black_box = False
                    black_box_content = ''
                    stored_predictions.clear()
                elif int(prediction[0]) == 2 and black_box_content:
                    threading.Thread(target=speak_labels, args=(stored_predictions,)).start()

            # Reset the gesture and time
            current_gesture = None
            gesture_start_time = None

    # Display the black box with stored predictions
    if display_black_box:
        cv2.rectangle(frame, (10, 10), (500, 100), (0, 0, 0), -1)
        cv2.putText(frame, black_box_content, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Hand Recognition', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        stored_predictions.pop()
        black_box_content = ' '.join(stored_predictions)
    elif key == ord('w'):
        stored_predictions.append('_')
        black_box_content = ' '.join(stored_predictions)

cap.release()
cv2.destroyAllWindows()
