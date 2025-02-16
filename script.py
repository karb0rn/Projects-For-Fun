import cv2

# Replace with your DroidCam IP and port
droidcam_url = "http://192.168.197.164:4747/video"
cap = cv2.VideoCapture(droidcam_url, cv2.CAP_FFMPEG)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Check your DroidCam connection.")
        break

    cv2.imshow('DroidCam Feed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()