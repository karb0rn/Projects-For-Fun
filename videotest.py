import cv2 as cv

cap = cv.VideoCapture(0)  # Open default webcam

while True:
    ret, frame = cap.read()  # Read a frame from the webcam

    if not ret:
        print("Failed to grab frame")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # Convert frame to grayscale

    cv.imshow('Color Frame', frame)  # Show original frame
    cv.imshow('Grayscale Frame', gray)  # Show grayscale frame

    if cv.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv.destroyAllWindows()
