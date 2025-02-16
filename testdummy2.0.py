import cv2
import numpy as np
import mediapipe as mp
import asyncio
import websockets
import json

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Start webcam feed
droidcam_url = "http://192.168.197.164:4747/video"
cap = cv2.VideoCapture(droidcam_url, cv2.CAP_FFMPEG)

def eye_aspect_ratio(eye_landmarks):
    # Extract (x, y) coordinates from NormalizedLandmark objects
    vertical1 = np.linalg.norm(np.array([eye_landmarks[1].x, eye_landmarks[1].y]) - np.array([eye_landmarks[0].x, eye_landmarks[0].y]))
    vertical2 = np.linalg.norm(np.array([eye_landmarks[3].x, eye_landmarks[3].y]) - np.array([eye_landmarks[2].x, eye_landmarks[2].y]))
    horizontal = np.linalg.norm(np.array([eye_landmarks[4].x, eye_landmarks[4].y]) - np.array([eye_landmarks[5].x, eye_landmarks[5].y]))
    if horizontal == 0:
        return 0  # Avoid division by zero
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

def mouth_aspect_ratio(mouth_landmarks):
    # Extract (x, y) coordinates from NormalizedLandmark objects
    vertical = np.linalg.norm(np.array([mouth_landmarks[1].x, mouth_landmarks[1].y]) - np.array([mouth_landmarks[0].x, mouth_landmarks[0].y]))
    horizontal = np.linalg.norm(np.array([mouth_landmarks[3].x, mouth_landmarks[3].y]) - np.array([mouth_landmarks[2].x, mouth_landmarks[2].y]))
    if horizontal == 0:
        return 0  # Avoid division by zero
    mar = vertical / horizontal
    return mar


EAR_THRESHOLD = 0.2  # Threshold for blink detection
MAR_THRESHOLD = 0.5  # Threshold for mouth open detection

# VTube Studio WebSocket URL
VTS_WS_URL = "ws://localhost:8001"

# Authentication token (if enabled in VTube Studio)
AUTH_TOKEN = "your_auth_token"

async def send_to_vtube_studio(websocket, ear_left, ear_right, mar):
    """
    Send EAR and MAR data to VTube Studio via WebSocket.
    """
    try:
        # Prepare the data to send
        expression_data = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "ExpressionRequest",
            "messageType": "ExpressionActivationRequest",
            "data": {
                "expressions": [
                    {"id": "EyeLeftX", "value": ear_left},  # Example parameter for left eye
                    {"id": "EyeRightX", "value": ear_right},  # Example parameter for right eye
                    {"id": "MouthOpen", "value": mar}  # Example parameter for mouth
                ]
            }
        }
        print("Sending data to VTube Studio:", expression_data)  # Debugging
        await websocket.send(json.dumps(expression_data))
    except Exception as e:
        print(f"Error sending data to VTube Studio: {e}")

async def connect_to_vtube_studio():
    """
    Connect to VTube Studio and send facial tracking data.
    """
    try:
        async with websockets.connect(VTS_WS_URL) as websocket:
            print("Connected to VTube Studio!")

            # Authenticate (if token is enabled)
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
                print("Authentication Response:", response)  # Debugging

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame. Check your DroidCam connection.")
                    break

                # Resize frame to reduce memory usage
                frame = cv2.resize(frame, (640, 480))

                # Convert the frame to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)

                # Convert back to BGR for display
                frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

                try:
                    if results.multi_face_landmarks:
                        print("Face detected!")
                        for face_landmarks in results.multi_face_landmarks:
                            if not face_landmarks.landmark:
                                print("No landmarks detected.")
                                continue

                            # Extract landmarks for eyes and mouth
                            left_eye_landmarks = [face_landmarks.landmark[i] for i in [33, 160, 158, 133, 153, 144]]
                            right_eye_landmarks = [face_landmarks.landmark[i] for i in [362, 385, 386, 263, 373, 380]]
                            mouth_landmarks = [face_landmarks.landmark[i] for i in [61, 291, 0, 17, 40, 270]]

                            # Draw landmarks on the frame (optional)
                            for landmark in left_eye_landmarks + right_eye_landmarks + mouth_landmarks:
                                x = int(landmark.x * frame.shape[1])
                                y = int(landmark.y * frame.shape[0])
                                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)  # Draw a green dot

                            # Detect blinks
                            left_ear = eye_aspect_ratio(left_eye_landmarks)
                            right_ear = eye_aspect_ratio(right_eye_landmarks)
                            print("Left EAR:", left_ear, "Right EAR:", right_ear)  # Debugging
                            if left_ear < EAR_THRESHOLD and right_ear < EAR_THRESHOLD:
                                print("Blink detected!")
                                cv2.putText(frame, "Blink Detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                            # Detect mouth open/close
                            mar = mouth_aspect_ratio(mouth_landmarks)
                            print("MAR:", mar)  # Debugging
                            if mar > MAR_THRESHOLD:
                                print("Mouth is open!")
                                cv2.putText(frame, "Mouth Open!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            else:
                                print("Mouth is closed.")

                            # Send EAR and MAR data to VTube Studio
                            await send_to_vtube_studio(websocket, left_ear, right_ear, mar)
                    else:
                        print("No face detected.")
                except Exception as e:
                    print(f"Error processing landmarks: {e}")

                # Show the frame
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