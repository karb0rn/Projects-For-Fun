import cv2
import numpy as np
import mediapipe as mp
import asyncio

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Start webcam feed
droidcam_url = "http://192.168.197.164:4747/video"
cap = cv2.VideoCapture(droidcam_url, cv2.CAP_FFMPEG)

# Thresholds
EAR_THRESHOLD = 0.22
MAR_THRESHOLD = 0.4

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
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Check your DroidCam connection.")
            break

        frame = cv2.resize(frame, (640, 480))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        width, height = frame.shape[1], frame.shape[0]

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

                # Blink Detection
                if left_ear < EAR_THRESHOLD and right_ear < EAR_THRESHOLD:
                    cv2.putText(frame, "Blink Detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Mouth Open Detection
                if mar > MAR_THRESHOLD:
                    cv2.putText(frame, "Mouth Open!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Send data to WebSocket
                await queue.put({"ear_left": left_ear, "ear_right": right_ear, "mar": mar})

        cv2.imshow("Facial Landmarks", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
