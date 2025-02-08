import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Constants
WINDOW_NAME = "Webcam Prediction"
CAPTURE_WIDTH, CAPTURE_HEIGHT = 640, 480
ROI_SCALE = 0.5  # 50% of the window
ROI_CLR = (0, 255, 0)
MODEL_PATH = 'models/rps_v01_56ep_0.9641acc_0.1089loss.h5'
CLASS_INDICES = {'paper': 0, 'rock': 1, 'scissors': 2}
CLASS_LABELS = {v: k for k, v in CLASS_INDICES.items()}
CLASS_IMG = {k: f"icons/{k}.png" for k in CLASS_INDICES}
PREDICTION_IMG_SIZE = (50, 50)
MODEL_INPUT_SIZE = (150, 150)

# Load Model
model = load_model(MODEL_PATH)
input_shape = model.input_shape[1:3]

# Access Webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
cv2.namedWindow(WINDOW_NAME)

# Define ROIs
roi_w, roi_h = int(CAPTURE_WIDTH * ROI_SCALE), int(CAPTURE_HEIGHT)
roi1_x, roi_y = 0, 0
roi2_x = CAPTURE_WIDTH // 2

def determine_winner(label1, label2):
    rules = {"rock": "scissors", "scissors": "paper", "paper": "rock"}
    if label1 == label2:
        return "Draw"
    elif rules.get(label1) == label2:
        return "Left Wins"
    else:
        return "Right Wins"

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to capture frame")
        break
    
    frame = cv2.flip(frame, 1)
    roi1 = frame[roi_y:roi_y+roi_h, roi1_x:roi1_x+roi_w]
    roi2 = frame[roi_y:roi_y+roi_h, roi2_x:roi2_x+roi_w]
    
    def process_roi(roi):
        roi_resized = cv2.resize(roi, MODEL_INPUT_SIZE, interpolation=cv2.INTER_AREA)
        roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB) / 255.0
        roi_expanded = np.expand_dims(roi_rgb, axis=0)
        
        prediction_prob = model.predict(roi_expanded, verbose=0)[0]
        predicted_class_index = np.argmax(prediction_prob)
        predicted_label = CLASS_LABELS[predicted_class_index]
        confidence = prediction_prob[predicted_class_index]
        
        return predicted_label, confidence
    
    label1, conf1 = process_roi(roi1)
    label2, conf2 = process_roi(roi2)
    winner = determine_winner(label1, label2)
    
    cv2.rectangle(frame, (roi1_x, roi_y), (roi1_x + roi_w, roi_y + roi_h), ROI_CLR, 2)
    cv2.putText(frame, f"{label1} ({conf1:.2f})", (roi1_x + 10, roi_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ROI_CLR, 2)
    
    cv2.rectangle(frame, (roi2_x, roi_y), (roi2_x + roi_w, roi_y + roi_h), ROI_CLR, 2)
    cv2.putText(frame, f"{label2} ({conf2:.2f})", (roi2_x + 10, roi_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ROI_CLR, 2)
    
    cv2.putText(frame, "HDU / Polytech - 2025", (CAPTURE_WIDTH//2 - 100, CAPTURE_HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, winner, (CAPTURE_WIDTH//2 - 60, CAPTURE_HEIGHT//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    cv2.imshow(WINDOW_NAME, frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()