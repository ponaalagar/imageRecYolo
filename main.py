import torch
import cv2
import numpy as np
from PIL import Image

def setup_model():
    """
    Load YOLOv5 model from torch hub
    """
    # Load YOLOv5 model from torch hub
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

def detect_and_label_image(image_path, confidence_threshold=0.5):
    """
    Detect objects in image and label them
    """
    # Load model
    model = setup_model()
    
    # Read image
    image = Image.open(image_path)
    
    # Perform detection
    results = model(image)
    
    # Convert PIL image to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Process detections
    detections = results.pandas().xyxy[0]  # Get detection results
    
    # Draw boxes and labels
    for idx, detection in detections.iterrows():
        if detection['confidence'] >= confidence_threshold:
            # Get coordinates
            x1, y1 = int(detection['xmin']), int(detection['ymin'])
            x2, y2 = int(detection['xmax']), int(detection['ymax'])
            
            # Get label and confidence
            label = f"{detection['name']} {detection['confidence']:.2f}"
            
            # Draw rectangle
            cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            cv2.putText(image_cv, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return image_cv

def main():
    """
    Main function to run the image detector
    """
    # Replace with your image path
    image_path = "E:\\imageRec\\pics\\apple-fruit.jpg"
    
    # Detect and label objects
    labeled_image = detect_and_label_image(image_path)
    
    # Display result
    cv2.imshow('Detected Objects', labeled_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save result
    cv2.imwrite('detected_objects.jpg', labeled_image)

if __name__ == "__main__":
    main()
