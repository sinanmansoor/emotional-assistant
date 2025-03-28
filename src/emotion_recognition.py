import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

class EmotionRecognizer:
    def __init__(self,
                 face_cascade_path='models/haarcascade_frontalface_default.xml',
                 emotion_model_path='models/emotion_model.h5'):
        # Load face detection model
        self.face_cascade = cv2.CascadeClassifier(face_cascade_path)

        # Load emotion classification model
        self.emotion_model = load_model(emotion_model_path)

        # Emotion labels
        self.emotion_labels = [
            'Angry', 'Disgust', 'Fear',
            'Happy', 'Sad', 'Surprise', 'Neutral'
        ]

    def detect_faces(self, frame):
        """
        Detect faces in a given frame

        Args:
            frame (numpy.ndarray): Input video frame

        Returns:
            list: Detected face regions
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        return faces

    def preprocess_face(self, face_img):
        """
        Preprocess face image for emotion classification

        Args:
            face_img (numpy.ndarray): Face image

        Returns:
            numpy.ndarray: Preprocessed face image
        """
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_img = cv2.resize(face_img, (48, 48))
        face_img = face_img / 255.0
        face_img = np.expand_dims(face_img, axis=0)
        face_img = np.expand_dims(face_img, axis=-1)
        return face_img

    def classify_emotion(self, face_img):
        """
        Classify emotion for a given face image

        Args:
            face_img (numpy.ndarray): Preprocessed face image

        Returns:
            str: Detected emotion label
        """
        prediction = self.emotion_model.predict(face_img)
        emotion_index = np.argmax(prediction)
        return self.emotion_labels[emotion_index]