import numpy as np
import cv2

def get_limits(color):
    # Convert BGR color to HSV
    c = np.uint8([[color]])
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)[0][0]

    # Ensure hue is within bounds (0 - 179)
    lower = np.array([max(hsvC[0] - 10, 0), 100, 100], dtype=np.uint8)
    upper = np.array([min(hsvC[0] + 10, 179), 255, 255], dtype=np.uint8)

    return lower, upper

# Define yellow in BGR
yellow = [0, 255, 255]

# Start webcam
webcam = cv2.VideoCapture(0)

while True:
    ret, frame = webcam.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lowerlimit, upperlimit = get_limits(color=yellow)
    mask = cv2.inRange(hsv_img, lowerlimit, upperlimit)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) > 500:  # Optional: ignore small areas
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow('Yellow Color Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
