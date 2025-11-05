import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from models import AlexNet  # We'll create this file for your model architecture

# Define transformations
transform = transforms.Compose([
    transforms.Resize((227, 227)),  # AlexNet input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model(model_path):
    """Load your trained PyTorch model"""
    model = AlexNet(num_classes=7)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model

def predict_emotion(model, input, emotion_labels, is_frame=False):
    if isinstance(input, str):  # Image file path
        image = Image.open(input).convert('RGB')
    else:  # Frame from webcam (numpy array)
        image = Image.fromarray(cv2.cvtColor(input, cv2.COLOR_BGR2RGB))
    
    # Preprocess the image
    image_tensor = transform(image).unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs.data, 1)
        emotion = emotion_labels[predicted.item()]
    
    # For webcam - return annotated frame and emotion
    if is_frame:
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.putText(frame, f"Emotion: {emotion}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 255, 0), 2)
        return frame, emotion
    
    # For image upload - just return emotion text
    return emotion