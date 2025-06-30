import cv2
import os
import pandas as pd

# Set the folder paths
video_folder_path = r'C:\ReactionTech\HML\H_data'
output_folder = r'C:\ReactionTech\HML\extracted_frames'

# Frame extraction rate (frames per second)
frames_per_second = 6 # Set this to how many frames you want to capture per second

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Known height (in meters) of the athlete in each video
athlete_heights = {
    'v_1.mov': 1.651,
    'v_2.mov': 1.651,
    'v_3.mp4': 1.778,
    'v_4.mp4': 1.499, 
    'v_5.mov': 1.499, 
    'v_6.mov': 1.499,
    'v_7.mov': 1.524,
    'v_8.mov': 1.524, 
    'v_9.mov': 1.575,
    'v_10.mov': 1.575,
    'v_11.mp4': 2.209,
    'v_12.mp4': 1.930,
    'v_13.mp4': 1.778,
    'v_14.mp4': 1.702,
    'v_15.mp4': 1.778,
    'v_16.mp4': 2.032,
    'v_17.mp4': 2.082,
    'v_18.mp4': 1.879,
    'v_19.mp4': 1.549,
    'v_20.mp4': 1.727,
    'v_21.mp4': 1.752,
    'v_22.mp4': 1.803,
    'v_23.mp4': 1.803,
    'v_24.mp4': 1.828,
    'v_25.mp4': 1.828,
    'v_26.mp4': 1.828,
    'v_27.mp4': 1.828,
    'v_28.mp4': 1.854,
    'v_29.mp4': 1.854,
    'v_30.mp4': 1.879,
    'v_31.mp4': 1.879,
    'v_32.mp4': 1.879,
    'v_33.mp4': 1.879,
    'v_34.mp4': 1.905,
    'v_35.mp4': 1.905,
    'v_36.mp4': 1.905,
    'v_37.mp4': 1.930,
    'v_38.mp4': 1.930,
    'v_39.mp4': 1.955,
    'v_40.mp4': 2.032,
    'v_41.mp4': 2.108,
    'v_42.mp4': 2.108,
    'v_43.mp4': 1.879,
    'v_44.mp4': 1.854,
    'v_45.mp4': 1.981,
    'v_46.mp4': 1.854,
    'v_47.mp4': 1.930,
    'v_48.mp4': 1.930,
    'v_49.mp4': 1.803,
    'v_50.mp4': 2.057,
    'v_51.mp4': 1.828,
    'v_52.mp4': 1.854,
    'v_53.mp4': 1.879,
    'v_54.mp4': 1.778,
    # Add more entries for each video and height
}
# Create a list to store height data for CSV
height_data = []

# Loop through each video in the folder
for video_name, height in athlete_heights.items():
    video_path = os.path.join(video_folder_path, video_name)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Could not open video {video_name}")
        continue

    # Get the video's FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / frames_per_second)  # Interval between frames to capture

    frame_count = 0
    saved_frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Save the frame if it matches the interval
        if frame_count % frame_interval == 0:
            # Define frame file path and save the frame
            frame_filename = f"{os.path.splitext(video_name)[0]}_frame{saved_frame_count}.jpg"
            frame_path = os.path.join(output_folder, frame_filename)
            cv2.imwrite(frame_path, frame)
            saved_frame_count += 1
            
            # Append height data to the list
            height_data.append({'frame': frame_filename, 'height': height})

        frame_count += 1

    cap.release()

# Save the height data to a CSV file
height_df = pd.DataFrame(height_data)
height_df.to_csv(os.path.join(output_folder, 'athlete_heights.csv'), index=False)

print("Frame extraction and height data storage completed.")
