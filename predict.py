import cv2
import numpy as np
import joblib

model = joblib.load("mood_model.pkl")
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def predict_mood(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48))
    img_flat = img.flatten().reshape(1, -1)
    prediction = model.predict(img_flat)
    return emotions[prediction[0]]
