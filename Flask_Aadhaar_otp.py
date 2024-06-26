import os
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model

# Initialize the Flask application
app = Flask(__name__)

# Load the pre-trained model
model = load_model('my_model.h5')

# Define characters and mappings
characters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
char_to_num = {char: idx for idx, char in enumerate(characters)}
num_to_char = {idx: char for char, idx in char_to_num.items()}

# Parameters
image_height = 50
image_width = 200
num_channels = 3
max_length = 5

# Function to preprocess image
def preprocess_image(image):
    image = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image")
    image = cv2.resize(image, (image_width, image_height))
    image = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(image, axis=0)

# Function to decode the predictions
def decode_prediction(pred):
    pred_text = []
    for idx in np.argmax(pred, axis=1):
        if idx != -1:
            pred_text.append(num_to_char.get(idx, ''))
        else:
            break
    return ''.join(pred_text)

# Define an endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    file.seek(0, os.SEEK_END)
    image_size = file.tell()
    file.seek(0)  # Reset the file pointer to the beginning of the file
    if image_size > 500 * 1024:  # Convert KB to bytes
        return jsonify({'prediction': "null",'responseMessage': 'File size exceeds.'}), 200
    
    # Check extension
    if not file.filename.endswith('.jpeg'):
        return jsonify({'prediction': "null",'responseMessage': 'Incorrect File Type.'}), 200

    # Make sure clientCode header is present and matches the expected value
    client_code = request.headers.get('clientCode')
    if client_code != 'Aadhaar_captcha':
        return jsonify({'prediction': "null",'responseMessage': 'Payload is Incorrect.'}), 200


    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 200

    try:
        image = preprocess_image(file)
        prediction = model.predict(image)
        pred_text = decode_prediction(prediction[0])
        return jsonify({'prediction': pred_text, 
                'responseMessage' : "Successfully Completed."}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 200

# Run the Flask application
if __name__ == '__main__':
    # app.run(debug=True, host='0.0.0.0', port=5000)
    app.run(debug=True, host='0.0.0.0', port=5000)