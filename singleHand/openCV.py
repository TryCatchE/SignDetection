import os
import cv2

# Directory where images will be saved
imgsDir = './images'

# Create the directory if it doesn't exist
if not os.path.exists(imgsDir):
    os.makedirs(imgsDir)

# Number of different image types to collect
imgType = 26

# Number of images to collect for each type
imgsNum = 100

# Open webcam for capturing images
cap = cv2.VideoCapture(0)

# Loop through each image type
for i in range(imgType):
    imgPath = os.path.join(imgsDir, str(i))

    # Create a subdirectory for each image type
    if not os.path.exists(imgPath):
        os.makedirs(imgPath)

    print(f'Collecting data for image type {i}')

    # Wait for user to press 'Q' to start capturing images
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    # Capture specified number of images for each type
    for counter in range(imgsNum):
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)

        # Save the captured image to the corresponding directory
        imgFile = os.path.join(imgPath, f'{counter}.jpg')
        cv2.imwrite(imgFile, frame)

# Close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
