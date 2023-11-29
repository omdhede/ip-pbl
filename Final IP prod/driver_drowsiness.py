# Import necessary libraries
import cv2
import numpy as np
import dlib
from imutils import face_utils
import pygame

# Initialize Pygame for sound
pygame.init()
# Load the alarm sound
sound = pygame.mixer.Sound("alarm.wav")

# Initialize face detector and shape predictor from dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize variables to track sleep, drowsy, and active states
sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)

# Function to compute Euclidean distance between two points
def compute(ptA, ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist

# Function to determine the blink status based on the eye landmarks
def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)

    if ratio > 0.25:
        return 2  # Both eyes closed
    elif 0.21 <= ratio <= 0.25:
        return 1  # One eye closed
    else:
        return 0  # Eyes open

# Open a video capture object
cap = cv2.VideoCapture(0)
while True:
    # Read a frame from the video capture
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)
    face_frame = np.zeros_like(frame)

    # Loop over detected faces
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        # Draw rectangle around the face
        face_frame = frame.copy()
        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Get facial landmarks
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        # Determine blink status for left and right eyes
        left_blink = blinked(landmarks[36], landmarks[37], landmarks[38],
                             landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43],
                              landmarks[44], landmarks[47], landmarks[46], landmarks[45])

        # Update sleep, drowsy, and active counters based on blink status
        if left_blink == 0 or right_blink == 0:
            sleep += 1
            drowsy = 0
            active = 0
            if sleep > 6:
                status = "SLEEPING !!!"
                color = (255, 0, 0)
                # Play the sound file
                sound.play()
        elif left_blink == 1 or right_blink == 1:
            sleep = 0
            active = 0
            drowsy += 1
            if drowsy > 6:
                status = "Drowsy!"
                color = (0, 0, 255)
                # Play the sound file
                sound.play()
        else:
            drowsy = 0
            sleep = 0
            active += 1
            if active > 6:
                status = "Active :)"
                color = (0, 255, 0)

        # Display status on the frame
        cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # Draw circles on the facial landmarks
        for n in range(0, 68):
            (x, y) = landmarks[n]
            cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

    # Display the original frame and the frame with detected face and landmarks
    cv2.imshow("Frame", frame)
    cv2.imshow("Result of detector", face_frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

# Release the video capture object
cap.release()
# Close all OpenCV windows
cv2.destroyAllWindows()
