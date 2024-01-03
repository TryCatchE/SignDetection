import os
import cv2

imgsDir = './doubleHands'
if not os.path.exists(imgsDir):
    os.makedirs(imgsDir)

imgType = 2
imgsNum = 50

cap = cv2.VideoCapture(0)

for i in range(imgType):
    imgPath = os.path.join(imgsDir, str(i))

    if not os.path.exists(imgPath):
        os.makedirs(imgPath)

    print(f'Collecting data for image type {i}')

    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    for counter in range(imgsNum):
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)

        imgFile = os.path.join(imgPath, f'{counter}.jpg')
        cv2.imwrite(imgFile, frame)

cap.release()
cv2.destroyAllWindows()
