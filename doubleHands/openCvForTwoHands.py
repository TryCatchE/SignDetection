import os
import cv2

# Directory to store captured images
imgsDir = './doubleHands'
if not os.path.exists(imgsDir):
    os.makedirs(imgsDir)

# Number of different hand gestures to capture
imgType = 3

# Number of images to capture for each gesture
imgsNum = 100

# Initialize webcam capture
cap = cv2.VideoCapture(0)

# Loop over each hand gesture type
for i in range(imgType):
    imgPath = os.path.join(imgsDir, str(i))

    # Create directory for each hand gesture type
    if not os.path.exists(imgPath):
        os.makedirs(imgPath)

    print(f'Collecting data for image type {i}')

    # Capture images
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    # Capture images for each hand gesture type
    for counter in range(imgsNum):
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)

        # Save captured image
        imgFile = os.path.join(imgPath, f'{counter}.jpg')
        cv2.imwrite(imgFile, frame)

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
