from flask import Flask, request, render_template, Response
import os
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from predict import predict_emotion, load_model

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load your PyTorch model
model = load_model('model/alexnet_fer2013_epoch15.pth')
model.eval()  # Set to evaluation mode

# Global variable for camera
camera = None
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def generate_frames():
    global camera
    camera = cv2.VideoCapture(0)  # 0 for default webcam
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Process frame for prediction
            processed_frame, emotion = predict_emotion(model, frame, emotion_labels, is_frame=True)
            
            # Encode frame for streaming
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html', prediction=None, image=None)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' in request.files:
        file = request.files['image']
        if file.filename != '':
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            result = predict_emotion(model, file_path, emotion_labels)
            return render_template('index.html', prediction=result, image=file.filename)
    return render_template('index.html', prediction=None, image=None)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_camera')
def stop_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None
    return "Camera stopped"

if __name__ == '__main__':
    app.run(debug=True)