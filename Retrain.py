import os
from glob import glob
from PIL import Image
import tensorflow as tf # type: ignore
import numpy as np # type: ignore
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

testdataX5 = []
img = Image.open("test/IMG_6917.webp").convert("RGB")
img = img.resize(targetsize)
img_array = np.array(img) / 255.0
testdataX5.append(img_array.flatten())
testdataX5 = np.array(testdataX5)

model = tf.keras.models.load_model('my_model.keras')
model.fit(finalImagesX , finalImagesAns, epochs=75)
print(model.predict(testdataX1)*100)
print(model.predict(testdataX2)*100)
print(model.predict(testdataX3)*100)
print(model.predict(testdataX4)*100)
print(model.predict(testdataX5)*100)
model.save('my_model3.keras')