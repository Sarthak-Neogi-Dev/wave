# Importing Packages
import pyautogui
import mediapipe as mp
import cv2
from mediapipe.tasks.python.vision import HandLandmarkerResult
import numpy as np

# Setting-up Variables
model_pipe = "models/hand_landmarker.task"
lastcallback = None
pointer_loc = None
pointer_loc_last = None
move_sens = 1.5
move_vector = None

# Setting-up Capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60)

# Setting-up Screen Size
screen_x, screen_y = pyautogui.size()

# Callback Function To Handle Result
def handle_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global lastcallback
    # print('hand landmarker result: {}'.format(result))
    lastcallback = result

# Setting-up Hand-Detector
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_pipe),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=handle_result,
    min_hand_detection_confidence = 0.2,
    min_hand_presence_confidence = 0.1,
    min_tracking_confidence = 0.1)

# Handling Capture Failure
if not cap.isOpened():
    print("Caputure can't be openned...")
    exit()

# Main Loop
with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read() # Reading Capture

        # Handling Error To Grab Frame
        if not ret:
            print("Error: Failed to grab frame.")
            break
        
        # Detecting Hand
        frame = cv2.flip(frame, 1)
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        landmarker.detect_async(mp_image, timestamp_ms)

        # Mouse Control
        mouse_loc = np.array(pyautogui.position())
        if lastcallback and lastcallback.hand_landmarks:
            if lastcallback.hand_landmarks[0]:
                index_finger = np.array([lastcallback.hand_landmarks[0][8].x,lastcallback.hand_landmarks[0][8].y])
                thumb_finger = np.array([lastcallback.hand_landmarks[0][4].x,lastcallback.hand_landmarks[0][4].y])
                middle_finger = np.array([lastcallback.hand_landmarks[0][12].x,lastcallback.hand_landmarks[0][12].y])
                
                move_dist = np.linalg.norm(index_finger - thumb_finger)
                click_dist = np.linalg.norm(middle_finger - thumb_finger)
                
                # Moving Mouse
                if move_dist < 0.03:
                    pointer_loc = np.array([lastcallback.hand_landmarks[0][8].x * screen_x,lastcallback.hand_landmarks[0][8].y * screen_y])
                    if pointer_loc_last is not None:
                        move_vector = (pointer_loc - pointer_loc_last) * move_sens
                        mouse_loc = mouse_loc + move_vector
                        pyautogui.moveTo(mouse_loc[0], mouse_loc[1], duration=0.1)
                    pointer_loc_last = pointer_loc
                else:
                    pointer_loc = None
                    pointer_loc_last = None

                # Clicking
                if click_dist < 0.03:
                    pyautogui.click(interval=0.1)
                
                print(mouse_loc[0], " : ",mouse_loc[1])

            else:
                pointer_loc = None
                pointer_loc_last = None

        # Displaying Webcam Feed
        # cv2.imshow('Webcam Feed', frame)

        # Handling Exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Clean-up
cap.release()
cv2.destroyAllWindows()