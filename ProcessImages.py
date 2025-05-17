import os
from glob import glob
from PIL import Image
import tensorflow as tf
import numpy as np
finalImagesAns = []
finalImagesX = []
targetsize = (224, 224)
image_folder = "/home/lloliva/coding/JamHacks2025/Organic"
image_files = glob(os.path.join(image_folder, "*.webp"))
Oragnic = np.array([])
for file in image_files:
    img = Image.open(file).convert("RGB")
    img = img.resize(targetsize)  
    img_array = np.array(img) / 255.0
    finalImagesX.append(img_array.flatten())
    finalImagesAns.append([1, 0, 0, 0])
image_folder = "/home/lloliva/coding/JamHacks2025/Paper"
image_files = glob(os.path.join(image_folder, "*.webp"))
Paper = np.array([])
for file in image_files:
    img = Image.open(file).convert("RGB")
    img = img.resize(targetsize)
    img_array = np.array(img) / 255.0
    finalImagesX.append(img_array.flatten())
    finalImagesAns.append([0, 1, 0, 0])

image_folder = "/home/lloliva/coding/JamHacks2025/PlasticCans"
image_files = glob(os.path.join(image_folder, "*.webp"))
PlasticCans = np.array([])
for file in image_files:
    img = Image.open(file).convert("RGB")
    img = img.resize(targetsize)
    img_array = np.array(img) / 255.0
    finalImagesX.append(img_array.flatten())
    finalImagesAns.append([0, 0, 1, 0])

image_folder = "/home/lloliva/coding/JamHacks2025/Organic"
image_files = glob(os.path.join(image_folder, "*.webp"))
Trash = np.array([])
for file in image_files:
    img = Image.open(file).convert("RGB")
    img = img.resize(targetsize)
    img_array = np.array(img) / 255.0
    finalImagesX.append(img_array.flatten())
    finalImagesAns.append([0, 0, 0, 1])
print(len(finalImagesX[0]))
print(len(finalImagesAns))
finalImagesX = np.array(finalImagesX)  # Convert list of arrays to 2D numpy array
print(finalImagesX.shape)  # Should be (num_samples, features_per_sample)
finalImagesAns = np.array(finalImagesAns)  # Convert list of arrays to 2D numpy array

testdataX1 = []
img = Image.open("test/IMG_2083.webp").convert("RGB")
img = img.resize(targetsize)
img_array = np.array(img) / 255.0
testdataX1.append(img_array.flatten())
testdataX1 = np.array(testdataX1)  # Convert list of arrays to 2D numpy array

testdataX2 = []
img = Image.open("test/IMG_6798.webp").convert("RGB")
img = img.resize(targetsize)
img_array = np.array(img) / 255.0
testdataX2.append(img_array.flatten())
testdataX2 = np.array(testdataX2)  # Convert list of arrays to 2D numpy array

testdataX3 = []
img = Image.open("test/IMG_6812.webp").convert("RGB")
img = img.resize(targetsize)
img_array = np.array(img) / 255.0
testdataX3.append(img_array.flatten())
testdataX3 = np.array(testdataX3)

testdataX4 = []
img = Image.open("test/IMG_6813.webp").convert("RGB")
img = img.resize(targetsize)
img_array = np.array(img) / 255.0
testdataX4.append(img_array.flatten())
testdataX4 = np.array(testdataX4)

model = tf.keras.Sequential([
    tf.keras.layers.Reshape((224, 224, 3), input_shape=(150528,)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])
model.compile(loss="categorical_crossentropy",optimizer="adam")
model.fit(finalImagesX , finalImagesAns, epochs=20)
eval = model.evaluate(finalImagesX,finalImagesAns)
print(model.predict(testdataX1))
print(model.predict(testdataX2))
print(model.predict(testdataX3))
print(model.predict(testdataX4))