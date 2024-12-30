import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import mediapipe as mp

# Parameters
gestures = ["thumbs_up", "thumbs_down", "open_palm", "peace_sign", "fist"]
base_output_path = "D:/SHUN/internship/Task4/gesture_dataset"  # Replace with the actual path
img_size = 64
num_classes = len(gestures)

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Step 1: Load and preprocess the dataset
def load_data():
    X, y = [], []
    for idx, gesture in enumerate(gestures):
        gesture_path = os.path.join(base_output_path, gesture)
        
        # Check if the gesture directory exists
        if not os.path.exists(gesture_path):
            print(f"Warning: Directory not found: {gesture_path}")
            continue
        
        # Process images in the gesture directory
        for img_name in os.listdir(gesture_path):
            img_path = os.path.join(gesture_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            
            if img is not None:  # Ensure the image is loaded successfully
                img = cv2.resize(img, (img_size, img_size))
                X.append(img)
                y.append(idx)
            else:
                print(f"Failed to load image: {img_path}")
    
    X = np.array(X, dtype="float32") / 255.0  # Normalize pixel values
    y = np.array(y)
    return X, y

# Load data and print sample counts
X, y = load_data()
print(f"Loaded dataset: {len(X)} images across {len(np.unique(y))} gestures.")

if len(X) == 0:
    raise ValueError("No data found. Please check dataset paths and structure.")

# One-hot encode labels
y = to_categorical(y, num_classes=num_classes)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Define the CNN model
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Create and train the model
model = create_model()
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# Save the model
model.save("gesture_recognition_model.h5")
print("Model saved successfully.")

# Step 4: Real-time Gesture Recognition using MediaPipe for hand detection
cap = cv2.VideoCapture(0)
print("Press 'q' to quit.")

def predict_gesture(frame):
    # Convert frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # If hand landmarks are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the bounding box of the hand
            h, w, _ = frame.shape
            x_min = int(min([landmark.x for landmark in hand_landmarks.landmark]) * w)
            y_min = int(min([landmark.y for landmark in hand_landmarks.landmark]) * h)
            x_max = int(max([landmark.x for landmark in hand_landmarks.landmark]) * w)
            y_max = int(max([landmark.y for landmark in hand_landmarks.landmark]) * h)

            # Ensure the bounding box is valid
            if x_max - x_min > 0 and y_max - y_min > 0:
                # Crop the hand region from the frame
                hand_crop = frame[y_min:y_max, x_min:x_max]
                hand_crop = cv2.resize(hand_crop, (img_size, img_size))

                # Predict the gesture
                img = np.expand_dims(hand_crop, axis=0) / 255.0
                predictions = model.predict(img)
                class_idx = np.argmax(predictions)
                return gestures[class_idx], frame, (x_min, y_min, x_max, y_max)

    return None, frame, None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from webcam.")
        break

    gesture, frame, bbox = predict_gesture(frame)
    
    if gesture:
        cv2.putText(frame, f"Gesture: {gesture}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        if bbox:
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Hand Gesture Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Gesture recognition complete!")
