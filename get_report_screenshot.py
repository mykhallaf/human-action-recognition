import cv2
import mediapipe as mp
import os
import random

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils

action_folder = "dataset/boxing"

if not os.path.exists(action_folder):
    print(f"Error: Folder '{action_folder}' not found.")
    exit()

video_files = [f for f in os.listdir(action_folder) if f.endswith('.avi')]
if not video_files:
    print("No .avi videos found!")
    exit()

video_path = os.path.join(action_folder, random.choice(video_files))
print(f"Processing: {video_path}")

cap = cv2.VideoCapture(video_path)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)

ret, frame = cap.read()

if ret:

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:

        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),  # Dot color
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)  # Line color
        )


        scale_factor = 4
        width = int(frame.shape[1] * scale_factor)
        height = int(frame.shape[0] * scale_factor)
        dim = (width, height)


        big_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_CUBIC)


        filename = "clean_report_screenshot.jpg"
        cv2.imwrite(filename, big_frame)
        print(f"Success! Saved BIG screenshot to: {filename}")


        cv2.imshow("Big Screenshot Preview", big_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No person detected in this frame. Try running again.")
else:
    print("Error reading video frame.")

cap.release()