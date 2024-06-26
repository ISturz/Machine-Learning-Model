import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import base64
from PIL import Image
from io import BytesIO
import torch


app = Flask(__name__)

import pickle
from model import proj2CNN  # Import the proj2CNN class from model.py

# Load the pre-trained model
with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_data = data.get('image_data', '')

        # Decode base64 image data and convert it to a NumPy array
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        image_array = np.array(image)

        # Convert the NumPy array to a PyTorch tensor
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0).float()

        # Perform prediction using the loaded model
        prediction = loaded_model.predict(image_tensor)

        print(jsonify(prediction.tolist()))
        return jsonify(prediction.tolist())
        
    except Exception as e:
        print("Error:", e)
        return jsonify(error="Error predicting."), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)
