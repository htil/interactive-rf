import time
import cv2
import mediapipe as mp
import numpy as np
from cam_utils.cam_helpers import predict_cam


model_path = 'cam_utils/gislr_64chess_acc87.47.h5'
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
is_draw = True
duration = 3
num_face_landmarks = 468
num_lhand_landmarks = 21
num_pose_landmarks = 33
num_rhand_landmarks = 21
lhand_offset = num_face_landmarks
pose_offset = lhand_offset + num_lhand_landmarks
rhand_offset = pose_offset + num_pose_landmarks

num_landmarks = num_face_landmarks + num_lhand_landmarks + \
                num_pose_landmarks + num_rhand_landmarks

frame_shape = (num_landmarks, 3)  # 3 for x, y, z
frames = []

# For webcam input:
cap = cv2.VideoCapture(0)
start_time = time.time()
with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        frame = np.empty(frame_shape)
        frame.fill(np.nan)

        if results.face_landmarks:
            frame[:num_face_landmarks] = np.array([np.array([l.x, l.y, l.z]) for l in results.face_landmarks.landmark])
        if results.left_hand_landmarks:
            frame[lhand_offset: pose_offset] = np.array([[l.x, l.y, l.z] for l in results.left_hand_landmarks.landmark])
        if results.pose_landmarks:
            frame[pose_offset: rhand_offset] = np.array([[l.x, l.y, l.z] for l in results.pose_landmarks.landmark])
        if results.right_hand_landmarks:
            frame[rhand_offset:] = np.array([[l.x, l.y, l.z] for l in results.right_hand_landmarks.landmark])
        frames.append(frame)
        # print(frame.shape)
        # break

        if is_draw:
            # Draw landmark annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles
                .get_default_pose_landmarks_style())
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))

        if time.time() - start_time >= duration:
            cv2.destroyAllWindows()
            frames = np.array(frames).astype(np.float32)
            sign = predict_cam(frames, model_path)
            print(sign)
            cv2.destroyAllWindows()
            break
        if cv2.waitKey(5) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
cap.release()
