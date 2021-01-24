from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# Establish optional arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default = "dataset", help="Path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="Path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type=str, default="mask_detector.model", help="Path to output face mask detector model")
args = vars(ap.parse_args())

# Parameters
INIT_LR = 1e-4 # Initial learning rate
EPOCHS = 20 # How many epochs to train for
BATCHSIZE = 32 # Batch Size

# Initial Setup
print("[INFO] Loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []





# Loop through images and append into data/labels
for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]
    
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)

    data.append(image)
    labels.append(label)

# Transform arrays to np arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

# Hot encoding labels so there is a value for both with_mask and without_mask
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# Splits into training and testing arrays
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

# Augments dataset with more translated images
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")





# Imports MobileNetV2
print("[INFO] Done loading images...")
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# Adds custom head
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7)) (headModel)
headModel = Flatten(name="flatten") (headModel)
headModel = Dense(128, activation="relu") (headModel)
headModel = Dropout(0.5) (headModel)
headModel = Dense(2, activation="softmax") (headModel)

# Combine base and head model
model = Model(inputs=baseModel.input, outputs=headModel)

# Freeze layers that belong to the baseModel
for layer in baseModel.layers:
    layer.trainable = False

# Compile the model
print("[INFO] Compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train the model
print("[INFO] Training model...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BATCHSIZE),
    steps_per_epoch=len(trainX) // BATCHSIZE,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BATCHSIZE,
    epochs=EPOCHS)

# Save the model
print("[INFO] Saving model...")
model.save(args["model"], save_format='h5')
print("[INFO] Model Saved...")





# Save plot of training
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])