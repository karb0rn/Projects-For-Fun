import cv2
import numpy as np
import mediapipe as mp
import asyncio
import websockets
import json
import threading
import time

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Webcam Setup
droidcam_url = "http://192.168.197.164:4747/video"
cap = cv2.VideoCapture(droidcam_url, cv2.CAP_FFMPEG)

# VTube Studio WebSocket
VTS_WS_URL = "ws://localhost:8001"
AUTH_TOKEN = "your_auth_token"

EAR_THRESHOLD = 0.2  
MAR_THRESHOLD = 0.5  
EYEBROW_THRESHOLD = 0.04  
HEAD_TILT_THRESHOLD = 0.02  
HEAD_NOD_THRESHOLD = 0.02  

latest_data = None
lock = threading.Lock()

def eye_aspect_ratio(eye_landmarks):
    vertical1 = np.linalg.norm(np.array([eye_landmarks[1].x, eye_landmarks[1].y]) - np.array([eye_landmarks[0].x, eye_landmarks[0].y]))
    vertical2 = np.linalg.norm(np.array([eye_landmarks[3].x, eye_landmarks[3].y]) - np.array([eye_landmarks[2].x, eye_landmarks[2].y]))
    horizontal = np.linalg.norm(np.array([eye_landmarks[4].x, eye_landmarks[4].y]) - np.array([eye_landmarks[5].x, eye_landmarks[5].y]))
    return (vertical1 + vertical2) / (2.0 * horizontal) if horizontal != 0 else 0  

def mouth_aspect_ratio(mouth_landmarks):
    vertical = np.linalg.norm(np.array([mouth_landmarks[1].x, mouth_landmarks[1].y]) - np.array([mouth_landmarks[0].x, mouth_landmarks[0].y]))
    horizontal = np.linalg.norm(np.array([mouth_landmarks[3].x, mouth_landmarks[3].y]) - np.array([mouth_landmarks[2].x, mouth_landmarks[2].y]))
    return vertical / horizontal if horizontal != 0 else 0  

def detect_head_tilt(nose, left_ear, right_ear):
    tilt = left_ear.y - right_ear.y
    if tilt > HEAD_TILT_THRESHOLD:
        return "Tilt Left"
    elif tilt < -HEAD_TILT_THRESHOLD:
        return "Tilt Right"
    return "Centered"

def detect_head_nod(nose, left_eye, right_eye):
    avg_eye_y = (left_eye.y + right_eye.y) / 2
    movement = avg_eye_y - nose.y
    if movement > HEAD_NOD_THRESHOLD:
        return "Nod Down"
    elif movement < -HEAD_NOD_THRESHOLD:
        return "Nod Up"
    return "Neutral"

def detect_eyebrow_raise(left_eyebrow, left_eye, right_eyebrow, right_eye):
    left_distance = left_eyebrow.y - left_eye.y
    right_distance = right_eyebrow.y - right_eye.y
    if left_distance > EYEBROW_THRESHOLD and right_distance > EYEBROW_THRESHOLD:
        return "Eyebrows Raised"
    return "Normal"

async def send_to_vtube_studio():
    global latest_data
    try:
        async with websockets.connect(VTS_WS_URL) as websocket:
            print("Connected to VTube Studio!")

            if AUTH_TOKEN:
                auth_message = {
                    "apiName": "VTubeStudioPublicAPI",
                    "apiVersion": "1.0",
                    "requestID": "AuthRequest",
                    "messageType": "AuthenticationRequest",
                    "data": {
                        "pluginName": "FacialTracker",
                        "pluginDeveloper": "YourName",
                        "authenticationToken": AUTH_TOKEN
                    }
                }
                await websocket.send(json.dumps(auth_message))
                response = await websocket.recv()
                print("Authentication Response:", response)

            while True:
                time.sleep(0.1)
                with lock:
                    if latest_data is None:
                        continue
                    data = latest_data.copy()
                    latest_data = None

                await websocket.send(json.dumps(data))
                print("Sent to VTube:", data)

    except Exception as e:
        print(f"WebSocket Error: {e}")

def process_video():
    global latest_data
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Check DroidCam connection.")
            break

        frame = cv2.resize(frame, (640, 480))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_eye = face_landmarks.landmark[33]
                right_eye = face_landmarks.landmark[362]
                left_ear = face_landmarks.landmark[234]
                right_ear = face_landmarks.landmark[454]
                nose = face_landmarks.landmark[1]
                left_eyebrow = face_landmarks.landmark[105]
                right_eyebrow = face_landmarks.landmark[334]

                left_ear_ratio = eye_aspect_ratio([face_landmarks.landmark[i] for i in [33, 160, 158, 133, 153, 144]])
                right_ear_ratio = eye_aspect_ratio([face_landmarks.landmark[i] for i in [362, 385, 386, 263, 373, 380]])
                mar = mouth_aspect_ratio([face_landmarks.landmark[i] for i in [61, 291, 0, 17, 40, 270]])

                head_tilt = detect_head_tilt(nose, left_ear, right_ear)
                head_nod = detect_head_nod(nose, left_eye, right_eye)
                eyebrow_state = detect_eyebrow_raise(left_eyebrow, left_eye, right_eyebrow, right_eye)

                with lock:
                    latest_data = {
                        "apiName": "VTubeStudioPublicAPI",
                        "apiVersion": "1.0",
                        "requestID": "ExpressionRequest",
                        "messageType": "ExpressionActivationRequest",
                        "data": {
                            "expressions": [
                                {"id": "EyeLeftX", "value": left_ear_ratio},
                                {"id": "EyeRightX", "value": right_ear_ratio},
                                {"id": "MouthOpen", "value": mar},
                                {"id": "EyebrowRaise", "value": 1 if eyebrow_state == "Eyebrows Raised" else 0},
                                {"id": "HeadTilt", "value": head_tilt},
                                {"id": "HeadNod", "value": head_nod}
                            ]
                        }
                    }

        cv2.imshow('Facial Landmarks', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "_main_":
    video_thread = threading.Thread(target=process_video)
    video_thread.start()
    asyncio.run(send_to_vtube_studio())