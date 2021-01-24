from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
from cv2 import cv2
import os

# Optional Arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, default="image.jpg", help="Path to input image")
ap.add_argument("-f", "--face",  type=str, default="face_detector", help="Path to face detector model directory")
ap.add_argument("-m", "--model", type=str, default="mask_detector.model", help="Path to trained face mask model")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="Minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Load face detector model
print("[INFO] Loading face detector model...")
protoxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"], "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(protoxtPath, weightsPath)

# Load face mask detector model
print("[INFO] Loading face mask detector model...")
model = load_model(args["model"])

# Load image
image = cv2.imread(args["image"])
orig = image.copy()
(h, w) = image.shape[:2]

# Construct blob for face model and pass through network
print("[INFO] Computing face detections...")
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104, 117, 123))
net.setInput(blob)
detections = net.forward()

# Loop over detections
for i in range(0, detections.shape[2]):
    # Extract all confidences from detections
    confidence = detections[0, 0, i, 2]

    # If the confidence is above the threshold, then it is a succesful detection of a face
    if confidence > args["confidence"]:
        # Finds relative postition/coords from the detection array, muitplies by image shape to find absolute coords
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # Double check that the boundaries fall within the image frame
        (startX, startY) = (max(0, startX), max(0, startY))
        (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

        # Attempt to find a mask within those image coords
        face = image[startY:endY, startX:endX]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB) # Ensure that the image data is stored as RGB
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        # Prediction
        (mask, withoutMask) = model.predict(face)[0]

        # Set label and color based on prediction
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # Draw label and rectangle
        cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

cv2.imshow("Output", image)
cv2.waitKey(0)