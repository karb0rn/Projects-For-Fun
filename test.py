import cv2
import numpy as np
import mediapipe as mp
import asyncio
import websockets
import json
import threading
import tkinter as tk

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Constants for thresholds
EAR_THRESHOLD = 0.2  # Blink threshold
MAR_THRESHOLD = 0.5  # Mouth open threshold
BROW_RAISE_THRESHOLD = 0.02  # Eyebrow raise threshold
HEAD_TILT_THRESHOLD = 10  # Tilt angle threshold

# WebSocket URL for VTube Studio
VTS_WS_URL = "ws://localhost:8001"
AUTH_TOKEN = "your_auth_token"

# DroidCam Stream URL
droidcam_url = "http://192.168.197.164:4747/video"

class FaceTracker:
    def _init_(self):
        self.cap = cv2.VideoCapture(droidcam_url, cv2.CAP_FFMPEG)
        self.frame_counter = 0
        self.PROCESS_EVERY_N_FRAMES = 3  # Skip frames for optimization
        self.websocket = None

        # Start WebSocket connection in a separate thread
        threading.Thread(target=asyncio.run, args=(self.connect_to_vtube_studio(),), daemon=True).start()

    def eye_aspect_ratio(self, eye_landmarks):
        vertical1 = np.linalg.norm(np.array([eye_landmarks[1].x, eye_landmarks[1].y]) - np.array([eye_landmarks[0].x, eye_landmarks[0].y]))
        vertical2 = np.linalg.norm(np.array([eye_landmarks[3].x, eye_landmarks[3].y]) - np.array([eye_landmarks[2].x, eye_landmarks[2].y]))
        horizontal = np.linalg.norm(np.array([eye_landmarks[4].x, eye_landmarks[4].y]) - np.array([eye_landmarks[5].x, eye_landmarks[5].y]))
        return 0 if horizontal == 0 else (vertical1 + vertical2) / (2.0 * horizontal)

    def mouth_aspect_ratio(self, mouth_landmarks):
        vertical = np.linalg.norm(np.array([mouth_landmarks[1].x, mouth_landmarks[1].y]) - np.array([mouth_landmarks[0].x, mouth_landmarks[0].y]))
        horizontal = np.linalg.norm(np.array([mouth_landmarks[3].x, mouth_landmarks[3].y]) - np.array([mouth_landmarks[2].x, mouth_landmarks[2].y]))
        return 0 if horizontal == 0 else vertical / horizontal

    def head_tilt_angle(self, face_landmarks, width, height):
        left_eye = face_landmarks.landmark[33]
        right_eye = face_landmarks.landmark[263]
        nose = face_landmarks.landmark[1]

        eye_center_x = (left_eye.x + right_eye.x) / 2
        eye_center_y = (left_eye.y + right_eye.y) / 2
        angle = np.arctan2(nose.y - eye_center_y, nose.x - eye_center_x) * (180.0 / np.pi)

        return angle

    async def connect_to_vtube_studio(self):
        while True:
            try:
                async with websockets.connect(VTS_WS_URL) as self.websocket:
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
                        await self.websocket.send(json.dumps(auth_message))
                        response = await self.websocket.recv()
                        print("Authentication Response:", response)

                    await self.process_video()

            except Exception as e:
                print(f"WebSocket Error: {e}, Reconnecting in 5 seconds...")
                await asyncio.sleep(5)

    async def send_to_vtube_studio(self, ear_left, ear_right, mar, head_tilt):
        if not self.websocket:
            return
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
                        {"id": "MouthOpen", "value": mar},
                        {"id": "HeadTilt", "value": head_tilt}
                    ]
                }
            }
            await self.websocket.send(json.dumps(expression_data))
        except Exception as e:
            print(f"Error sending data to VTube Studio: {e}")

    async def process_video(self):
        while self.cap.isOpened():
            self.frame_counter += 1
            ret, frame = self.cap.read()
            if not ret or self.frame_counter % self.PROCESS_EVERY_N_FRAMES != 0:
                continue

            frame = cv2.resize(frame, (640, 480))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    left_eye = [face_landmarks.landmark[i] for i in [33, 160, 158, 133, 153, 144]]
                    right_eye = [face_landmarks.landmark[i] for i in [362, 385, 386, 263, 373, 380]]
                    mouth = [face_landmarks.landmark[i] for i in [61, 291, 0, 17, 40, 270]]

                    left_ear = self.eye_aspect_ratio(left_eye)
                    right_ear = self.eye_aspect_ratio(right_eye)
                    mar = self.mouth_aspect_ratio(mouth)
                    head_tilt = self.head_tilt_angle(face_landmarks, frame.shape[1], frame.shape[0])

                    print(f"EAR Left: {left_ear}, EAR Right: {right_ear}, MAR: {mar}, Head Tilt: {head_tilt}")

                    if left_ear < EAR_THRESHOLD and right_ear < EAR_THRESHOLD:
                        print("Blink detected!")
                    if left_ear < EAR_THRESHOLD < right_ear:
                        print("Left wink detected!")
                    if right_ear < EAR_THRESHOLD < left_ear:
                        print("Right wink detected!")
                    if mar > MAR_THRESHOLD:
                        print("Mouth Open!")

                    await self.send_to_vtube_studio(left_ear, right_ear, mar, head_tilt)

            cv2.imshow('Facial Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if _name_ == "_main_":
    tracker = FaceTracker()
    asyncio.run(tracker.connect_to_vtube_studio())