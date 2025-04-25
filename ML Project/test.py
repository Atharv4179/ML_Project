import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import Counter

# === CONFIGURATION ===
MODEL_PATH = r"C:\Users\chinc\Desktop\Ai Powered Virtual Riffle Shooting Coach\model_output\best_model.h5"  # Path of the model
CONFIDENCE_THRESHOLD = 0.6
CATEGORIES = ['Good with kit', 'Good without kit', 'Bad with kit', 'Bad without kit', 'Relaxed']

# === Load Model ===
model = tf.keras.models.load_model(MODEL_PATH)

# === Mediapipe Pose ===
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# === Feedback Function ===
def analyze_posture(landmarks, prediction_idx):
    feedback = []
    if prediction_idx in [2, 3]:  # Only bad postures
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        left_foot = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value]
        right_foot = landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value]

        if left_elbow.y < left_shoulder.y - 0.05 or right_elbow.y < right_shoulder.y - 0.05:
            feedback.append("Elbow is raised - relax it")

        hip_center_x = (left_hip.x + right_hip.x) / 2
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        if abs(hip_center_x - shoulder_center_x) > 0.05:
            feedback.append("Waist is bent - center your body")

        if abs(left_foot.y - right_foot.y) > 0.03:
            feedback.append("Feet are not parallel")

        shoulder_distance = np.sqrt(
            (right_shoulder.x - left_shoulder.x) ** 2 +
            (right_shoulder.y - left_shoulder.y) ** 2
        )
        if shoulder_distance < 0.25:
            feedback.append("Shoulders compressed - relax")

    return feedback

# === Extract keypoints ===
def extract_keypoints(landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()

# === Image Inference ===
with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
    # Assuming you load the image path from elsewhere, e.g., user input or predefined image
    image_path = input("Please enter the image path: ")  # Enter the image path when prompted
    image = cv2.imread(image_path)

    if image is None:
        print(f"Could not read the image.")
    else:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if not results.pose_landmarks:
            print("No pose detected in the image.")
        else:
            landmarks = results.pose_landmarks.landmark
            keypoints = extract_keypoints(landmarks)
            input_data = np.expand_dims(keypoints, axis=0)

            prediction = model.predict(input_data, verbose=0)[0]
            prediction_idx = np.argmax(prediction)
            confidence = prediction[prediction_idx]

            label = f"{CATEGORIES[prediction_idx]} ({confidence:.2f})"
            feedback = analyze_posture(landmarks, prediction_idx) if confidence > CONFIDENCE_THRESHOLD else []

            # Draw pose
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Draw label
            cv2.putText(image, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Feedback
            for i, line in enumerate(feedback):
                y = 70 + i * 30
                cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Show image
            cv2.imshow("Posture Feedback", image)
            cv2.waitKey(0)

cv2.destroyAllWindows()
