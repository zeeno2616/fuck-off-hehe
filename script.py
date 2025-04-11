import cv2
import mediapipe as mp
import os
import time

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Check if the middle finger is up and others are down
def is_middle_finger_only_up(hand_landmarks):
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]

    fingers_up = []
    for tip, pip in zip(finger_tips, finger_pips):
        tip_y = hand_landmarks.landmark[tip].y
        pip_y = hand_landmarks.landmark[pip].y
        fingers_up.append(tip_y < pip_y)  # If tip is above pip, the finger is up

    return fingers_up == [False, True, False, False]

# Start capturing webcam
cap = cv2.VideoCapture(0)

shutdown_triggered = False

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip the frame horizontally and convert to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if is_middle_finger_only_up(hand_landmarks):
                cv2.putText(frame, "Middle Finger Detected! Shutting down...", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                shutdown_triggered = True
                break

    cv2.imshow("Gesture Detector", frame)

    if shutdown_triggered:
        time.sleep(2)
        os.system("shutdown /s /t 1")  #SHUTDOWN THE DAMN THINGGGGGGG
        break

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
