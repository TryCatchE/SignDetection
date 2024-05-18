import pickle
import cv2
import mediapipe as mediap
import numpy as np
import pyttsx3
import threading
import string
import time


# Load the saved models
words_model = pickle.load(open('./models/words_model.p', 'rb'))  # Adjust path as needed
model_single_hand_dir = pickle.load(open('./models/alphabet_model.p', 'rb'))  # Adjust path as needed
model_double_hands_dir = pickle.load(open('./models/interactions_model.p', 'rb'))  # Adjust path as needed

words_model = words_model['model']
model_single_hand = model_single_hand_dir['model']
model_double_hands = model_double_hands_dir['model']

prev_label = None  # Variable to store the previous prediction label
label_start_time = None  # Time when the current label was first detected
stored_predictions = []  # Array to store predictions
display_black_box = False  # Flag to control the display of the black box
black_box_content = ""  # Content to display in the black box
last_stored_prediction = None  # Variable to store the last stored prediction
last_stored_time = None  # Time when the last prediction was stored
store_interval = 5  # Interval in seconds to store new predictions
current_gesture = None# New variables for gesture tracking
gesture_start_time = None
PREDICTION_INTERVAL = 1.5  # Time in seconds to wait before making a prediction
wordsModel = False

# Initialize Text-to-Speech engine
engine = pyttsx3.init()
engine_busy = False  # Flag to track if the engine is currently speaking

cap = cv2.VideoCapture(0)

hands = mediap.solutions.hands
drawUtls = mediap.solutions.drawing_utils
styleUtls = mediap.solutions.drawing_styles

handsObj = hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3)
# Define specific words for certain keys
specific_labels = {
    0: 'Thank you',
    1: 'Love you',
    2: 'Yes',
    3: 'No',
    4: 'Eat',
    5: 'Drink',
    6: 'Agree',
    7: 'Question'
}

# Define default value for other keys
default_label = 'None'

# Create the labels dictionary with specific keys mapped to words and default value for others
labels_words = {key: specific_labels.get(key, default_label) for key in range(26)}
labels_single_hand = {i: letter for i, letter in enumerate(string.ascii_uppercase)}  # Adjust labels as needed
labels_double_hands = {0: 'open', 1: 'talk', 2: 'remove',3: 'turn off', 4: 'change', 5: 'space'}  # Adjust labels as needed


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
    
    mode_text = "Letters" if not wordsModel else "Words"
    cv2.putText(frame, mode_text, (W - 90, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

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
                
        if wordsModel:
            if num_hands == 1:
                model = words_model
                labels = labels_words
                expected_num_features = 42  # Adjust as per your single hand model
            else:
                model = model_double_hands
                labels = labels_double_hands
                expected_num_features = 84 
        else:
            # Decide between single hand or double hands model based on the number of hands detected
            if num_hands == 1:
                model = model_single_hand
                labels = labels_single_hand
                expected_num_features = 42  # Adjust as per your single hand model
            else:  # Assuming 2 hands
                model = model_double_hands
                labels = labels_double_hands
                expected_num_features = 84 

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
                black_box_content = ''.join(stored_predictions)
                
                if num_hands == 1 and wordsModel:
                    print("detecting words model")
                    
                    if int(prediction[0]) == 1:
                        # wordsModel = False
                        print("made words model to false")
            #_TODO IF NEEDED 
            if num_hands == 2 :
            # if num_hands == 2 and not wordsModel:
                if int(prediction[0]) == 0:
                    display_black_box = True
                elif int(prediction[0]) == 1 and black_box_content:
                    # thats the way to add the words when we train teh model ___TODO!
                    # stored_predictions.append(predLabel)
                    # black_box_content = ''.join(stored_predictions)
                    threading.Thread(target=speak_labels, args=(stored_predictions,)).start()
                elif int(prediction[0]) == 2 and len(stored_predictions) > 0:
                    stored_predictions.pop()
                    black_box_content = ''.join(stored_predictions)
                elif int(prediction[0]) == 3:
                    display_black_box = False
                    black_box_content = ''
                    stored_predictions.clear()
                elif int(prediction[0]) == 4:
                    wordsModel = not wordsModel
                elif int(prediction[0]) == 5:
                    stored_predictions.append(' ')
                    black_box_content = ''.join(stored_predictions) 
                    
            # if num_hands == 2 and wordsModel:
            #     print("2 hand words")

            # Reset the gesture and time
            current_gesture = None
            gesture_start_time = None

    # Display the black box with stored predictions
    if display_black_box:
        cv2.rectangle(frame, (10, 10), (W - 10, 100), (0, 0, 0), -1)
        black_box_content ='Hello World'
        cv2.putText(frame, black_box_content, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('Hand Recognition', frame)
    key = cv2.waitKey(1) & 0xFF

cap.release()
cv2.destroyAllWindows()
