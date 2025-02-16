import cv2
import numpy as np
import mediapipe as mp

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
                print("Left EAR:", left_ear, "Right EAR:", right_ear)
                if left_ear < 0.2 and right_ear < 0.2:
                    print("Blink detected!")

                # Detect mouth open/close
                mar = mouth_aspect_ratio(mouth_landmarks)
                print("MAR:", mar)
                if mar > 0.5:
                    print("Mouth is open!")
                else:
                    print("Mouth is closed.")
        else:
            print("No face detected.")
    except Exception as e:
        print(f"Error processing landmarks: {e}")

    # Show the frame
    cv2.imshow('Facial Landmarks', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()