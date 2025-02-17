import cv2
import numpy as np
import mediapipe as mp
import asyncio
import websockets
import json

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Start webcam feed
droidcam_url = "http://192.168.197.164:4747/video"
cap = cv2.VideoCapture(droidcam_url, cv2.CAP_FFMPEG)

def calculate_distance(landmark1, landmark2, width, height):
    """
    Calculate Euclidean distance between two facial landmarks.
    Convert normalized coordinates (0-1) to pixel-based coordinates.
    """
    x1, y1 = int(landmark1.x * width), int(landmark1.y * height)
    x2, y2 = int(landmark2.x * width), int(landmark2.y * height)
    return np.linalg.norm([x2 - x1, y2 - y1])

def eye_aspect_ratio(eye_landmarks, width, height):
    """
    Compute the Eye Aspect Ratio (EAR) to detect blinks.
    """
    vertical1 = calculate_distance(eye_landmarks[1], eye_landmarks[5], width, height)
    vertical2 = calculate_distance(eye_landmarks[2], eye_landmarks[4], width, height)
    horizontal = calculate_distance(eye_landmarks[0], eye_landmarks[3], width, height)
    
    if horizontal == 0:
        return 0  # Avoid division by zero
    return (vertical1 + vertical2) / (2.0 * horizontal)

def mouth_aspect_ratio(mouth_landmarks, width, height):
    """
    Compute the Mouth Aspect Ratio (MAR) to detect mouth open status.
    """
    vertical = calculate_distance(mouth_landmarks[2], mouth_landmarks[3], width, height)
    horizontal = calculate_distance(mouth_landmarks[0], mouth_landmarks[1], width, height)
    
    if horizontal == 0:
        return 0  # Avoid division by zero
    return vertical / horizontal

# Thresholds
EAR_THRESHOLD = 0.22  # Blink detection
MAR_THRESHOLD = 0.4   # Mouth open detection

# WebSocket URL for VTube Studio
VTS_WS_URL = "ws://localhost:8001"
AUTH_TOKEN = "your_auth_token"

async def send_to_vtube_studio(websocket, ear_left, ear_right, mar):
    """Send EAR and MAR data to VTube Studio."""
    try:
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
    except Exception as e:
        print(f"Error sending data to VTube Studio: {e}")

async def connect_to_vtube_studio():
    """Connect to VTube Studio and process facial tracking data."""
    try:
        async with websockets.connect(VTS_WS_URL) as websocket:
            print("Connected to VTube Studio!")

            # Authentication
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
                        # Eye landmarks
                        left_eye_indices = [33, 160, 158, 133, 153, 144]
                        right_eye_indices = [362, 385, 386, 263, 373, 380]
                        mouth_indices = [78, 308, 13, 14]  # Adjusted for better accuracy

                        left_eye_landmarks = [face_landmarks.landmark[i] for i in left_eye_indices]
                        right_eye_landmarks = [face_landmarks.landmark[i] for i in right_eye_indices]
                        mouth_landmarks = [face_landmarks.landmark[i] for i in mouth_indices]

                        # Calculate EAR and MAR
                        left_ear = eye_aspect_ratio(left_eye_landmarks, width, height)
                        right_ear = eye_aspect_ratio(right_eye_landmarks, width, height)
                        mar = mouth_aspect_ratio(mouth_landmarks, width, height)

                        # Blink Detection
                        if left_ear < EAR_THRESHOLD and right_ear < EAR_THRESHOLD:
                            cv2.putText(frame, "Blink Detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        # Mouth Open Detection
                        if mar > MAR_THRESHOLD:
                            cv2.putText(frame, "Mouth Open!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        # Send data to VTube Studio
                        await send_to_vtube_studio(websocket, left_ear, right_ear, mar)

                cv2.imshow('Facial Landmarks', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except Exception as e:
        print(f"WebSocket connection error: {e}")

# Run the WebSocket client
asyncio.get_event_loop().run_until_complete(connect_to_vtube_studio())

# Release resources
cap.release()
cv2.destroyAllWindows()