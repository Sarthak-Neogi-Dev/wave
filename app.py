# Importing Packages
import pyautogui
import pyautogui
import mediapipe as mp
import cv2
from mediapipe.tasks.python.vision import HandLandmarkerResult

# Setting-up Variables
model_pipe = "models/hand_landmarker.task"
lastcallback = None

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
    min_hand_detection_confidence = 0.5,
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

        #Moving Mouse
        if lastcallback and lastcallback.hand_landmarks:
            if lastcallback.hand_landmarks[0]:
                mouse_x = lastcallback.hand_landmarks[0][8].x * screen_x
                mouse_y = lastcallback.hand_landmarks[0][8].y * screen_y
                pyautogui.moveTo(mouse_x,mouse_y)
                print(mouse_x," : ",mouse_y)

        # Displaying Webcam Feed
        cv2.imshow('Webcam Feed', frame)

        # Handling Exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Clean-up
cap.release()
cv2.destroyAllWindows()