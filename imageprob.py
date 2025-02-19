from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import cv2
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def setup_model():
    """
    Load YOLOv5 model
    """
    model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)
    model.conf = 0.25  # Lower confidence threshold to catch more potential matches
    return model

def calculate_probabilities(image, product_names):
    """
    Calculate probability scores for specific product names in the image
    """
    model = setup_model()
    
    # Perform detection
    results = model(image)
    
    # Process detections
    detections = results.pandas().xyxy[0]
    
    # Initialize probabilities dictionary
    probabilities = {name: 0.0 for name in product_names}
    
    # Calculate probabilities for each product name
    for idx, detection in detections.iterrows():
        detected_name = detection['name']
        confidence = float(detection['confidence'])
        
        # Check if the detected object matches any of our product names
        for product_name in product_names:
            # Case insensitive comparison
            if detected_name.lower() == product_name.lower():
                # Update probability if new confidence is higher
                probabilities[product_name] = max(probabilities[product_name], confidence)
            # Check for partial matches (e.g., "coffee mug" matches "mug")
            elif detected_name.lower() in product_name.lower() or product_name.lower() in detected_name.lower():
                # For partial matches, we reduce the confidence slightly
                adjusted_confidence = confidence * 0.8
                probabilities[product_name] = max(probabilities[product_name], adjusted_confidence)
    
    return probabilities

@app.route('/calculate_probability', methods=['POST'])
def calculate_image_probabilities():
    """
    API endpoint to calculate probabilities for specific products in an uploaded image
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    if 'product_names' not in request.form:
        return jsonify({'error': 'No product names provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Get product names from the request
        product_names = request.form['product_names'].split(',')
        product_names = [name.strip() for name in product_names]
        
        # Read image
        image = Image.open(io.BytesIO(file.read()))
        
        # Calculate probabilities
        probabilities = calculate_probabilities(image, product_names)
        
        # Format response
        response = {
            'probabilities': {
                name: round(prob * 100, 2) for name, prob in probabilities.items()
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(port=5001, debug=True)