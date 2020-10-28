from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream

import numpy as np
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, face_net, mask_net):
    # get frame dimensions and create a blob from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

    # passing blob through network and optaining face detection
    face_net.setInput(blob)
    detections = face_net.forward()
    print(detections.shape)

    # initializing list of faces, their corresponding locations, list of predictions from face mask model
    faces = []
    locations = []
    preds = []

    # looping over the detections
    for i in range(0, detections.shape[2]):
        # extracting the respective probability associated with the detection
        proba = detections[0, 0, i, 2]

        # filtering out weak detections by setting a threshold
        if proba > 0.5:
            # compute (x, y)-coordinates for box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (X_start, y_start, X_end, y_end) = box.astype("int")

            # ensuring the box lies within the dimensions of the frame
            (X_start, y_start) = (max(0, X_start), max(0, y_start))
            (X_end, y_end) = (min(w - 1, X_end), min(h - 1, y_end))

            # extracting the face ROI and converting it from BGR to RGB channel
            # ordering, resizing (to 224 x 244) and further preprocessing
            face = frame[y_start:y_end, X_start:X_end]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locations.append((X_start, y_start, X_end, y_end))
    
    # theshold to only make predicitons when a face is detected
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = mask_net.predict(faces, batch_size=32)

    # return a tuple of the face location and their respective prediction from the model
    return (locations, preds)

# loading serialized face detector model (pretrained from GitHub)
prototxtPath = f"face_detector/deploy.prototxt"
weightsPath = f"face_detector/res10_300x300_ssd_iter_140000.caffemodel"
face_net = cv2.dnn.readNet(prototxtPath, weightsPath)

# laoding the face mask detector model
mask_net = load_model("face_mask_detector.model")

# initialize video stream to detect faces
print("[Info] starting video stream to show bubu model ...")
vs = VideoStream(src=0).start()

# looping over frames of video stream
while True:
    # resize frame from threaded video stream to a w of 400px
    frame = vs.read()
    frame = imutils.resize(frame, width=800)

    # detect faces and give probability of them wearing a face mask
    (locations, preds) = detect_and_predict_mask(frame, face_net, mask_net)

    # loop over face locations and respective probabilities
    for (box, prediction) in zip(locations, preds):
        (X_start, y_start, X_end, y_end) = box
        (has_mask, no_mask) = prediction

        # set class label and color of box border
        label = "Mask" if has_mask > no_mask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # including the respective probability in box label
        label = "{}: {:.2f}%".format(label, max(has_mask, no_mask)*100)

        # displaying label and box on output frame
        cv2.putText(frame, label, (X_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (X_start, y_start), (X_end, y_end), color, 2)

    # show output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if "q"-key is pressed, break from the loop
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()