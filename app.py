from flask import Flask, request, jsonify, send_file
from flask_cors import CORS  # Add this import
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import os

model = tf.keras.models.load_model('my_model7.keras')
targetsize = (224, 224)
app = Flask(__name__)
CORS(app)  # Add this line to enable CORS

@app.route('/api/classify', methods=['POST'])
def classify_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        # Read the image directly from the request
        image = request.files['image']
        img = Image.open(io.BytesIO(image.read())).convert("RGB")
        img = img.resize(targetsize)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array.reshape((1, -1))  

        # Predict using the model
        predictions = model.predict(img_array)
        labels = ["Organic", "Paper", "Plastic/Cans", "Trash"]
        prediction_dict = {labels[i]: round(float(predictions[0][i])) for i in range(len(labels))}
        
        return jsonify({
            'predictions': prediction_dict,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return send_file('website.html')

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'status': 'working'})

if __name__ == '__main__':
    app.run(debug=True)