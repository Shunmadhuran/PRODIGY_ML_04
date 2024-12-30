import cv2
import os
import mediapipe as mp

# Define gesture names and output paths
gestures = ["thumbs_up", "thumbs_down", "open_palm", "peace_sign", "fist"]
base_output_path = "gesture_dataset"

# Create directories for each gesture
for gesture in gestures:
    os.makedirs(os.path.join(base_output_path, gesture), exist_ok=True)

# Initialize webcam and MediaPipe Hand module
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

image_count = 0

print("Press 'q' to quit at any time.")
while True:
    # Ask the user to input the gesture they will show
    print(f"Please show the gesture: {gestures}")
    user_gesture = input("Enter the gesture name (e.g., thumbs_up): ").strip()

    if user_gesture not in gestures:
        print("Invalid gesture. Please try again.")
        continue

    print(f"Now, show your {user_gesture} gesture.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for better user experience
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and detect hands
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            # For each hand found, extract the bounding box and save the hand image
            for hand_landmarks in results.multi_hand_landmarks:
                # Get the bounding box of the hand
                h, w, _ = frame.shape
                x_min = int(min([landmark.x for landmark in hand_landmarks.landmark]) * w)
                y_min = int(min([landmark.y for landmark in hand_landmarks.landmark]) * h)
                x_max = int(max([landmark.x for landmark in hand_landmarks.landmark]) * w)
                y_max = int(max([landmark.y for landmark in hand_landmarks.landmark]) * h)

                # Crop the hand region from the frame
                hand_crop = frame[y_min:y_max, x_min:x_max]

                # Show the cropped hand region on the frame
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, f"Showing {user_gesture}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # Capture the cropped hand image on pressing 'c'
                if cv2.waitKey(1) & 0xFF == ord('c'):
                    output_path = os.path.join(base_output_path, user_gesture)
                    img_name = f"{user_gesture}_{image_count}.jpg"
                    img_path = os.path.join(output_path, img_name)
                    cv2.imwrite(img_path, hand_crop)
                    image_count += 1
                    print(f"Captured {img_name} for gesture {user_gesture}")

        # Show the live feed with detected hands
        cv2.imshow("Hand Gesture Dataset Generator", frame)

        # Quit the script on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Ask the user if they want to continue capturing gestures
    continue_capture = input("Do you want to capture another gesture? (y/n): ").strip().lower()
    if continue_capture != 'y':
        break

cap.release()
cv2.destroyAllWindows()

print("Dataset generation complete!")
