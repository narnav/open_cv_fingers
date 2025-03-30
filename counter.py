import cv2
import mediapipe as mp

# Initialize Mediapipe Hand Module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Open Webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for a mirrored view
    frame = cv2.flip(frame, 1)

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect Hands
    result = hands.process(rgb_frame)
    
    # Draw Landmarks and Count Fingers
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get landmark positions
            landmark_list = [(lm.x, lm.y) for lm in hand_landmarks.landmark]

            # Define fingers to check (Tip & Lower Joint)
            fingers = [
                (4, 2),  # Thumb
                (8, 6),  # Index
                (12, 10),  # Middle
                (16, 14),  # Ring
                (20, 18)   # Pinky
            ]

            # Count fingers up
            fingers_up = sum(landmark_list[tip][1] < landmark_list[joint][1] for tip, joint in fingers)
            fingers_down = 5 - fingers_up  # Total fingers - Up fingers

            # Display count on frame
            cv2.putText(frame, f'Fingers Up: {fingers_up}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Fingers Down: {fingers_down}', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display Frame
    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
