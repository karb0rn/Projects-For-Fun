import cv2
import numpy as np
import mediapipe as mp
import asyncio
import websockets
import json
import time
import threading

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Start webcam feed
DROIDCAM_URL = "http://192.168.14.180:4747/video"
cap = cv2.VideoCapture(DROIDCAM_URL, cv2.CAP_FFMPEG)

# Thresholds
EAR_THRESHOLD = 0.22  # Blink detection
MAR_THRESHOLD = 0.4   # Mouth open detection
VTS_WS_URL = "ws://localhost:8001"
AUTH_TOKEN = "your_auth_token"

# FPS Calculation
frame_count = 0
start_time = time.time()

# WebSocket Connection
async def websocket_task():
    global ear_left, ear_right, mar
    while True:
        try:
            async with websockets.connect(VTS_WS_URL) as websocket:
                print("Connected to VTube Studio!")
                
                # Authenticate
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
                    if ear_left is not None and ear_right is not None and mar is not None:
                        expression_data = {
                            "apiName": "VTubeStudioPublicAPI",
                            "apiVersion": "1.0",
                            "requestID": "ExpressionRequest",
                            "messageType": "ExpressionActivationRequest",
                            "data": {
                                "expressions": [
                                    {"id": "EyeLeftX", "value": ear_left},
                                    {"id": "EyeRightX", "value": ear_right},
                                    {"id": "MouthOpen", "value": mar}
                                ]
                            }
                        }
                        await websocket.send(json.dumps(expression_data))
                    await asyncio.sleep(0.05)
        except Exception as e:
            print(f"WebSocket connection error: {e}, retrying in 3 seconds...")
            await asyncio.sleep(3)

# Start WebSocket in a separate thread
def start_websocket():
    asyncio.run(websocket_task())

threading.Thread(target=start_websocket, daemon=True).start()

def process_frames():
    global ear_left, ear_right, mar, frame_count, start_time
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Check DroidCam connection.")
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
                
                def calculate_distance(landmark1, landmark2):
                    x1, y1 = int(landmark1.x * width), int(landmark1.y * height)
                    x2, y2 = int(landmark2.x * width), int(landmark2.y * height)
                    return np.linalg.norm([x2 - x1, y2 - y1])

                def eye_aspect_ratio(eye_landmarks):
                    vertical1 = calculate_distance(eye_landmarks[1], eye_landmarks[5])
                    vertical2 = calculate_distance(eye_landmarks[2], eye_landmarks[4])
                    horizontal = calculate_distance(eye_landmarks[0], eye_landmarks[3])
                    return (vertical1 + vertical2) / (2.0 * horizontal) if horizontal else 0

                def mouth_aspect_ratio(mouth_landmarks):
                    vertical = calculate_distance(mouth_landmarks[2], mouth_landmarks[3])
                    horizontal = calculate_distance(mouth_landmarks[0], mouth_landmarks[1])
                    return vertical / horizontal if horizontal else 0

                ear_left = eye_aspect_ratio(left_eye_landmarks)
                ear_right = eye_aspect_ratio(right_eye_landmarks)
                mar = mouth_aspect_ratio(mouth_landmarks)

                if ear_left < EAR_THRESHOLD and ear_right < EAR_THRESHOLD:
                    cv2.putText(frame, "Blink Detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if mar > MAR_THRESHOLD:
                    cv2.putText(frame, "Mouth Open!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.putText(frame, f"EAR: {ear_left:.2f}, {ear_right:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"MAR: {mar:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow('Facial Landmarks', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Start Frame Processing
process_frames()
cap.release()
cv2.destroyAllWindows()