import requests

url = 'http://localhost:8000/predict'  # Correct endpoint for predictions
data = {'image_data': 'base64_encoded_image_data_here'}  # Include your image data here

r = requests.post(url, json=data)

print(r.json())