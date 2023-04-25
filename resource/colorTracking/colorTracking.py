# Importing all modules
import cv2
import numpy as np
import time

# Specifying upper and lower ranges of color to detect in hsv format
lower = np.array([0, 0, 0])
upper = np.array([255, 0, 0]) # (These ranges will detect Yellow)

# Capturing webcam footage
# webcam_video = cv2.VideoCapture(1)
cap = cv2.VideoCapture("./data/videos/colorSwing.mp4")

while cap.isOpened():
    ret, frame = cap.read() # Reading webcam footage

    # img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Converting BGR image to HSV format

    mask = cv2.inRange(frame, lower, upper) # Masking the image to find our color

    mask_contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Finding contours in mask image

    # Finding position of all contours
    if len(mask_contours) != 0:
        for mask_contour in mask_contours:
            if cv2.contourArea(mask_contour) > 500:
                x, y, w, h = cv2.boundingRect(mask_contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3) #drawing rectangle

    cv2.imshow("mask image", mask) # Displaying mask image

    cv2.imshow("window", frame) # Displaying webcam image
    time.sleep(0.1)

    cv2.waitKey(1)