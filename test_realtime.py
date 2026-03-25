import cv2
import mediapipe as mp
import numpy as np
import pickle
import pandas as pd
import warnings


warnings.filterwarnings("ignore")


print("Loading...")
try:
    with open('action_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('selector.pkl', 'rb') as f:
        selector = pickle.load(f)
    print("Loaded successfully!")
except FileNotFoundError:
    print("Error: Files not found.")
    exit()


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam started. Press 'q' to quit.")


window_name = 'Real-time Test'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break


    frame = cv2.flip(frame, 1)


    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:

        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)


        row = []
        for lm in results.pose_landmarks.landmark:
            row.extend([lm.x, lm.y, lm.z, lm.visibility])


        X = pd.DataFrame([row])

        try:

            X_selected = selector.transform(X)


            classes = ['boxing', 'hand_clapping', 'hand_waving', 'jogging', 'running', 'walking']


            prediction_index = model.predict(X_selected)[0]
            action_name = classes[prediction_index]


            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_selected)[0]
                conf = max(proba)


                if conf > 0.70:
                    display_text = f"{action_name} ({conf * 100:.0f}%)"
                    text_color = (255, 255, 255)

                    box_color = (0, 200, 0)
                else:
                    display_text = "Unsure..."
                    text_color = (200, 200, 200)
                    box_color = (245, 117, 16)
            else:
                display_text = action_name
                text_color = (255, 255, 255)
                box_color = (245, 117, 16)


            cv2.rectangle(frame, (0, 0), (350, 60), box_color, -1)


            cv2.putText(frame, 'ACTION', (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


            cv2.putText(frame, display_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)

        except Exception as e:

            print(f"Error: {e}")
            pass

    cv2.imshow(window_name, frame)


    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()