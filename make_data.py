import mediapipe
print(f"I am loading mediapipe from: {mediapipe.__file__}")
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)


landmark_names = []
for i in range(33):
    landmark_names.extend([f'x_{i}', f'y_{i}', f'z_{i}', f'v_{i}'])
landmark_names.append('label')


data = []
dataset_path = "dataset"
actions = ["boxing", "hand_clapping", "hand_waving", "jogging", "running", "walking"]


print("Starting Feature Extraction... this might take a while.")

for action in actions:
    folder_path = os.path.join(dataset_path, action)

    if not os.path.exists(folder_path):
        print(f"WARNING: Folder {folder_path} not found!")
        continue


    video_files = os.listdir(folder_path)
    count = 0

    for video_name in video_files:

        print(f"Processing {action} video {count + 1}/{len(video_files)}: {video_name}")

        video_path = os.path.join(folder_path, video_name)
        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break


            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                row = []
                for lm in results.pose_landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z, lm.visibility])

                row.append(action)
                data.append(row)
        cap.release()
        count += 1

    print(f"Finished processing ALL videos for {action}")



df = pd.DataFrame(data, columns=landmark_names)
df.to_csv('action_dataset.csv', index=False)
print("Data saved to action_dataset.csv")