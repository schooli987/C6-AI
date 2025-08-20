import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()

# Open webcam
cap = cv2.VideoCapture(0)

# Function to map fingers to numbers using if conditions
def fingers_up(hand_landmarks):
    tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
    fingers = []

    # Thumb (assuming right hand)
    if hand_landmarks.landmark[tips[0]].x < hand_landmarks.landmark[tips[0]-1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers
    for i in range(1, 5):
        if hand_landmarks.landmark[tips[i]].y < hand_landmarks.landmark[tips[i]-2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    # Convert to number using if conditions (AFTER all fingers are added)
    if fingers == [0,1,0,0,0]:
        return 1
    elif fingers == [0,1,1,0,0]:
        return 2
    elif fingers == [0,1,1,1,0]:
        return 3
    elif fingers == [0,1,1,1,1]:
        return 4
    elif fingers == [1,0,0,0,0]:
        return 5
    elif fingers == [1,1,0,0,0]:
        return 6
    elif fingers == [1,1,1,0,0]:
        return 7
    elif fingers == [1,1,1,1,0]:
        return 8
    elif fingers == [1,1,1,1,1]:
        return 9
    else:
        return 0  # no valid number detected

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    finger_count = 0
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            finger_count = fingers_up(hand_landmarks)

    # Display detected number
    cv2.putText(frame, f"The number is: {finger_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Recognition", frame)

    if cv2.waitKey(1) & 0xFF ==  ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
