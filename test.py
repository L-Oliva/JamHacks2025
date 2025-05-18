from PIL import Image
import tensorflow as tf # type: ignore
import numpy as np # type: ignore
targetsize = (224, 224)
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
img = Image.open("test/IMG_6914.webp").convert("RGB")
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

model = tf.keras.models.load_model('my_model7.keras')
print(model.predict(testdataX1)*100)
print(model.predict(testdataX2)*100)
print(model.predict(testdataX3)*100)
print(model.predict(testdataX4)*100)
print(model.predict(testdataX5)*100)