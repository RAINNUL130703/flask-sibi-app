from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import base64
import tensorflow as tf
from io import BytesIO
import mediapipe as mp
import cv2
import os
import time
import json

# Load class mapping dari file JSON
with open("class_mapping.json", "r") as f:
    CLASS_MAPPING = json.load(f)

app = Flask(__name__)

# Konfigurasi
DEBUG_MODE = True
MODEL_PATH = 'model_SIBI.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# Inisialisasi MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def draw_hand_annotations(image, hand_landmarks):
    annotated_image = image.copy()
    mp_drawing.draw_landmarks(annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    x_coords = [lm.x * image.shape[1] for lm in hand_landmarks.landmark]
    y_coords = [lm.y * image.shape[0] for lm in hand_landmarks.landmark]
    x_min, x_max = int(min(x_coords)), int(max(x_coords))
    y_min, y_max = int(min(y_coords)), int(max(y_coords))
    cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    return annotated_image, (x_min, y_min, x_max, y_max)

def extract_feature_from_image(pil_image):
    try:
        image = np.array(pil_image)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        results = hands.process(image_rgb)
        if not results.multi_hand_landmarks:
            return None, None, None

        hand_landmarks = results.multi_hand_landmarks[0]
        annotated_image, bbox = draw_hand_annotations(image_bgr, hand_landmarks)

        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        landmarks = np.array(landmarks).reshape(1, 63, 1).astype(np.float32)
        landmarks = (landmarks - np.mean(landmarks)) / np.std(landmarks)

        return landmarks, annotated_image, bbox

    except Exception as e:
        print(f"Error saat memproses gambar: {str(e)}")
        return None, None, None

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        data = request.get_json()
        img_bytes = base64.b64decode(data['image'].split(',')[1])
        pil_image = Image.open(BytesIO(img_bytes)).convert('RGB')

        input_tensor, annotated_image, bbox = extract_feature_from_image(pil_image)

        if input_tensor is None:
            return jsonify({'error': 'No hand detected'}), 400

        prediction = model.predict(input_tensor)
        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        predicted_letter = CLASS_MAPPING.get(str(predicted_class), "Unknown")

        if DEBUG_MODE:
            debug_path = f"debug/annotated_{int(time.time())}.jpg"
            cv2.imwrite(debug_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

        return jsonify({
            'predicted_class': predicted_class,
            'predicted_letter': predicted_letter,
            'confidence': confidence,
            'bbox': bbox,
            'annotated_image': base64.b64encode(cv2.imencode('.jpg', annotated_image)[1]).decode()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
