from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
from cv2 import cv2
import os
import imutils
from imutils.video import VideoStream
import time

# Optional arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str, default="face_detector", help="Path to face detector model directory")
ap.add_argument("-m", "--model", type=str, default="mask_detector.model", help="Path to trained face mask model")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="Minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Detects Mask for each frame
def detect_mask(frame, faceNet, maskNet):
    # Takes frame shape and passes it into the network
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 117, 123))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # Create arrays for later
    faces = []
    locs = []
    preds = []

    # For every detection, extract coords and face image
    for i in range(0, detections.shape[2]):
        # Extract confidences=
        confidence = detections[0, 0, i, 2]

        # Filter out weak confidences
        if confidence > args["confidence"]:
            # Transforms relative box coordinates to frame coordinates
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensure coords are within frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w-1, endX), min(h-1, endY))

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            # face = np.expand_dims(face, axis=0)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # Only want to predict if at least one face is detected
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)

# Load face detector
print("[INFO] Loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Load mask detector
print("[INFO] Loading mask detector model...")
maskNet = load_model(args["model"])

# Start the video stream and wait 2 seconds for it to start
print("Beginning video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Loop over frames
while True:
    # Load frame
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # Pass frame and models into func
    (locs, preds) = detect_mask(frame, faceNet, maskNet)

    # Loop over detected face locations
    for (box, pred) in zip(locs, preds):
        # Unpack
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # Assign label and color
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # Display probability
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # Draw label and rectangle
        cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # Show frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0XFF

    # Break from stream if "q" is pressed
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()