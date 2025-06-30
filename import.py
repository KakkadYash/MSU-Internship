import pandas as pd
import os
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from kerastuner import HyperModel, RandomSearch

EarlyStopping = tf.keras.callbacks.EarlyStopping
ReduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau
data_path = r'C:\ReactionTech\HML\extracted_frames'
csv_file_path = r'C:\ReactionTech\HML\extracted_frames\athlete_heights.csv'
data = pd.read_csv(csv_file_path)

def extract_landmarks(image):
    """Extract body landmarks using MediaPipe Pose."""
    mp_pose = mp.solutions.pose
    with mp_pose.Pose() as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            landmarks = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
            return np.array(landmarks).flatten()  # Flatten the 3D coordinates into a 1D array
    return np.zeros(99)  # Return zero array if no landmarks are detected

def calculate_additional_features(landmarks):
    """Calculate additional features based on landmarks."""
    # Example: Calculate distance between specific landmarks
    distance_nose_shoulder = np.linalg.norm(landmarks[0:3] - landmarks[3:6])  # 3D distance
    return np.array([distance_nose_shoulder])

def load_images_and_heights(data, image_folder):
    images = []
    heights = []
    for _, row in data.iterrows():
        image_path = os.path.join(image_folder, row['frame'])
        image = cv2.imread(image_path)
        if image is not None:
            landmarks = extract_landmarks(image)
            additional_features = calculate_additional_features(landmarks)
            combined_features = np.concatenate((landmarks, additional_features))  # Combine original landmarks and new features
            images.append(combined_features)
            heights.append(row['height'])
    return np.array(images), np.array(heights)

# Load data
X, y = load_images_and_heights(data, data_path)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

class HeightEstimationModel(HyperModel):
    def build(self, hp):
        model = keras.models.Sequential()
        model.add(keras.layers.Input(shape=(100,)))  # 99 landmarks + 1 additional feature

        # Reduce the number of layers and units
        for i in range(hp.Int('num_layers', 1, 2)):  # Use 1-2 layers instead of up to 4
            model.add(keras.layers.Dense(units=hp.Int('units_' + str(i), 32, 128, step=32), activation='relu'))
            model.add(keras.layers.Dropout(0.3))  # Keep dropout to prevent overfitting
        
        model.add(keras.layers.Dense(1))  # Output layer to predict height
        model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', [1e-3, 1e-4])),  # Reduce learning rate range
                      loss='mean_squared_error',
                      metrics=['mae'])
        return model

# Set up the tuner
tuner = RandomSearch(
    HeightEstimationModel(),
    objective='val_mae',
    max_trials=10,
    executions_per_trial=1,
    directory='my_dir',
    project_name='height_estimation_tuning'
)

# Run the hyperparameter search
tuner.search(X_train, y_train, epochs=50, validation_data=(X_val, y_val))

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Learning rate reduction
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

# Train the best model with callbacks
history = best_model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), 
                          batch_size=32, callbacks=[early_stopping, reduce_lr])

# Save the model after training
best_model.save('height_estimation_model.h5')

# Load the saved model
model = keras.models.load_model('height_estimation_model.h5')


# Function to process the video and predict height
def predict_height(video_path):
    video_capture = cv2.VideoCapture(video_path)
    all_landmarks = []
    frame_count = 0
    batch_size = 5  # Number of frames to process at once
    
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        # Sample every nth frame (e.g., every 5th frame)
        if frame_count % batch_size == 0:
            # Extract landmarks for the current frame
            landmarks = extract_landmarks(frame)
            additional_features = calculate_additional_features(landmarks)
            combined_features = np.concatenate((landmarks, additional_features))  # Combine original landmarks and new features
            all_landmarks.append(combined_features)
        
        frame_count += 1

    video_capture.release()

    # Convert the list of landmarks to a NumPy array
    all_landmarks = np.array(all_landmarks)

    # Remove frames with zero landmarks
    valid_landmarks = all_landmarks[~np.all(all_landmarks == 0, axis=1)]

    if valid_landmarks.size == 0:
        print("No valid landmarks detected.")
        return None

    # Optionally average the landmarks before prediction for speed-up
    mean_landmarks = np.mean(valid_landmarks, axis=0).reshape(1, -1)

    # Predict height using the model
    predicted_height = model.predict(mean_landmarks)[0][0]  # Single prediction

    return predicted_height

# Test the model with a new video
new_video_path = r"C:\ReactionTech\Data\presentation\video.MOV" # Replace with the path to your new video
estimated_height = predict_height(new_video_path)

if estimated_height is not None:
    print(f"Estimated height of the athlete: {estimated_height:.2f} meters")