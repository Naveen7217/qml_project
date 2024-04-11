# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import imutils
import cv2

# prompt the user to input the paths
cascade_path = input("Enter the path to where the face cascade resides: ")
model_path = input("Enter the path to the pre-trained smile detector: ")
video_path = input("Enter the path to the (optional) video file: ")

# load the face detector cascade and smile detector CNN
detector = cv2.CascadeClassifier(cascade_path)
model = load_model(model_path)

# if a video path was not supplied, grab the references to the webcam
if not video_path:
    print('[INFO] starting video capture...')
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(video_path)

# keep looping
while True:
    # grab the current frame
    (grabbed, frame) = camera.read()

    # if we are viewing a video and we did not grab a frame, then we
    # have reached the end of the video
    if video_path and not grabbed:
        break

    # resize the frame, convert it to grayscale, and then clone the
    # original frame so we draw on it later in the program
    frame = imutils.resize(frame, width=700)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_clone = frame.copy()

    # detect faces in the input frame, then clone the frame so that we can draw on it
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for (fX, fY, fW, fH) in rects:
        # extract the ROI of the face from the grayscale image
        # resize it to a fixed 28x28 pixels, and then prepare the
        # ROI for classification via the CNN
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (28, 28))
        roi = roi.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # determine the probabilities of both 'smiling' and 'not smiling',
        # then set the label accordingly
        (not_smiling, smiling) = model.predict(roi)[0]
        if 0.4 < smiling < 0.7:
            label = 'Fake Smile'
        else:
            label = 'Smiling' if smiling > not_smiling else "Not Smiling"

        # display the label and bounding box on the output frame
        if label == 'Smiling':
            cv2.putText(frame_clone, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            cv2.rectangle(frame_clone, (fX, fY), (fX + fW, fY + fH), (0, 255, 0), 2)
        elif label == 'Fake Smile':
            cv2.putText(frame_clone, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 2)
            cv2.rectangle(frame_clone, (fX, fY), (fX + fW, fY + fH), (0, 165, 255), 2)
        else:
            cv2.putText(frame_clone, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(frame_clone, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)

    # show our detected face along with smiling/not smiling labels
    cv2.imshow('Face', frame_clone)

    # if 'q' key is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
