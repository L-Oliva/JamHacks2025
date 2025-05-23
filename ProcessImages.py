import os
from glob import glob
from PIL import Image
import tensorflow as tf
import numpy as np
finalImagesAns = []
finalImagesX = []
targetsize = (224, 224)
folders = ["Organic","Paper","PlasticCans","Trash"]
for i in range(len(folders)):
    image_folder = "/home/lloliva/coding/JamHacks2025/" + folders[i]
    image_files = glob(os.path.join(image_folder, "*.webp"))
    Oragnic = np.array([])
    for file in image_files:
        img = Image.open(file).convert("RGB")
        img = img.resize(targetsize)  
        img_array = np.array(img) / 255.0
        finalImagesX.append(img_array.flatten())
        if i == 0:
            finalImagesAns.append([1, 0, 0, 0])
        elif i == 1:
            finalImagesAns.append([0, 1, 0, 0])
        elif i == 2:
            finalImagesAns.append([0, 0, 1, 0])
        else:
            finalImagesAns.append([0, 0, 0, 1])
finalImagesX = np.array(finalImagesX)  # Convert list of arrays to 2D numpy array
finalImagesAns = np.array(finalImagesAns)  # Convert list of arrays to 2D numpy array

testdataX1 = []
img = Image.open("test/IMG_2083.webp").convert("RGB")
img = img.resize(targetsize)
img_array = np.array(img) / 255.0
testdataX1.append(img_array.flatten())
testdataX1 = np.array(testdataX1)  # Convert list of arrays to 2D numpy array

testdataX2 = []
img = Image.open("test/IMG_6812.webp").convert("RGB")
img = img.resize(targetsize)
img_array = np.array(img) / 255.0
testdataX2.append(img_array.flatten())
testdataX2 = np.array(testdataX2)  # Convert list of arrays to 2D numpy array

testdataX3 = []
img = Image.open("test/IMG_6913.webp").convert("RGB")
img = img.resize(targetsize)
img_array = np.array(img) / 255.0
testdataX3.append(img_array.flatten())
testdataX3 = np.array(testdataX3)

testdataX4 = []
img = Image.open("test/IMG_6968 (2).webp").convert("RGB")
img = img.resize(targetsize)
img_array = np.array(img) / 255.0
testdataX4.append(img_array.flatten())
testdataX4 = np.array(testdataX4)

testdataX5 = []
img = Image.open("test/PXL_20250517_222705654.webp").convert("RGB")
img = img.resize(targetsize)
img_array = np.array(img) / 255.0
testdataX5.append(img_array.flatten())
testdataX5 = np.array(testdataX5)

model = tf.keras.Sequential([
    tf.keras.layers.Reshape((224, 224, 3), input_shape=(150528,)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(200, activation='relu'),
    tf.keras.layers.Dense(136, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])
model.compile(loss="categorical_crossentropy",optimizer="adam")
model.fit(finalImagesX , finalImagesAns, epochs=45)
eval = model.evaluate(finalImagesX,finalImagesAns)
print(model.predict(testdataX1)*100)
print(model.predict(testdataX2)*100)
print(model.predict(testdataX3)*100)
print(model.predict(testdataX4)*100)
print(model.predict(testdataX5)*100)
model.save('my_model7.keras')