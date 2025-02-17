import cv2
import numpy as np
import mediapipe as mp
import asyncio
from collections import deque

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Start webcam feed
DROIDCAM_URL = ""
cap = cv2.VideoCapture(DROIDCAM_URL, cv2.CAP_FFMPEG)

# Thresholds
EAR_THRESHOLD = 0.22
MAR_THRESHOLD = 0.4

# Moving average window
SMOOTHING_WINDOW = 5
ears_left = deque(maxlen=SMOOTHING_WINDOW)
ears_right = deque(maxlen=SMOOTHING_WINDOW)
mars = deque(maxlen=SMOOTHING_WINDOW)

def calculate_distance(landmark1, landmark2, width, height):
    """Calculate Euclidean distance between two landmarks in pixel coordinates."""
    x1, y1 = int(landmark1.x * width), int(landmark1.y * height)
    x2, y2 = int(landmark2.x * width), int(landmark2.y * height)
    return np.linalg.norm([x2 - x1, y2 - y1])

def eye_aspect_ratio(eye_landmarks, width, height):
    """Compute the Eye Aspect Ratio (EAR)."""
    vertical1 = calculate_distance(eye_landmarks[1], eye_landmarks[5], width, height)
    vertical2 = calculate_distance(eye_landmarks[2], eye_landmarks[4], width, height)
    horizontal = calculate_distance(eye_landmarks[0], eye_landmarks[3], width, height)
    return (vertical1 + vertical2) / (2.0 * horizontal) if horizontal != 0 else 0

def mouth_aspect_ratio(mouth_landmarks, width, height):
    """Compute the Mouth Aspect Ratio (MAR)."""
    vertical = calculate_distance(mouth_landmarks[2], mouth_landmarks[3], width, height)
    horizontal = calculate_distance(mouth_landmarks[0], mouth_landmarks[1], width, height)
    return vertical / horizontal if horizontal != 0 else 0

async def process_facial_tracking(queue):
    """Process video frames and compute EAR and MAR values."""
    global cap
    while True:
        if not cap.isOpened():
            print("Reconnecting to DroidCam...")
            cap = cv2.VideoCapture(DROIDCAM_URL, cv2.CAP_FFMPEG)
            await asyncio.sleep(1)
            continue

        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Check your DroidCam connection.")
            await asyncio.sleep(1)
            continue

        frame = cv2.resize(frame, (480, 360))  # Lower resolution for speed
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        width, height = frame.shape[1], frame.shape[0]

        try:
            results = face_mesh.process(rgb_frame)
        except Exception as e:
            print("Error processing frame:", e)
            continue

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_eye_indices = [33, 160, 158, 133, 153, 144]
                right_eye_indices = [362, 385, 386, 263, 373, 380]
                mouth_indices = [78, 308, 13, 14]

                left_eye_landmarks = [face_landmarks.landmark[i] for i in left_eye_indices]
                right_eye_landmarks = [face_landmarks.landmark[i] for i in right_eye_indices]
                mouth_landmarks = [face_landmarks.landmark[i] for i in mouth_indices]

                left_ear = eye_aspect_ratio(left_eye_landmarks, width, height)
                right_ear = eye_aspect_ratio(right_eye_landmarks, width, height)
                mar = mouth_aspect_ratio(mouth_landmarks, width, height)

                # Apply moving average smoothing
                ears_left.append(left_ear)
                ears_right.append(right_ear)
                mars.append(mar)
                smooth_left_ear = np.mean(ears_left)
                smooth_right_ear = np.mean(ears_right)
                smooth_mar = np.mean(mars)

                # Blink Detection
                if smooth_left_ear < EAR_THRESHOLD and smooth_right_ear < EAR_THRESHOLD:
                    cv2.putText(frame, "Blink Detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Mouth Open Detection
                if smooth_mar > MAR_THRESHOLD:
                    cv2.putText(frame, "Mouth Open!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Send data to WebSocket queue
                await queue.put({"ear_left": smooth_left_ear, "ear_right": smooth_right_ear, "mar": smooth_mar})

        cv2.imshow("Facial Landmarks", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        await asyncio.sleep(0.01)  # Non-blocking sleep for async handling

    cap.release()
    cv2.destroyAllWindows()
