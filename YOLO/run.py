from ultralytics import YOLO
import cv2
import numpy as np

# Load your two models
model_gray = YOLO("final/final3_gray/weights/best.pt")   # trained on grayscale
model_style = YOLO("final/final4_style/weights/best.pt")       # trained on color (or another model)

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Prepare grayscale version for gray model ---
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)  # 3 channels

    # --- Run both models ---
    results_gray = model_gray(gray_frame, stream=True)
    results_style = model_style(frame, stream=True)

    # --- Copy frames for display ---
    display_gray = gray_frame.copy()
    display_style = frame.copy()

    # Draw results for grayscale model (green boxes)
    for r in results_gray:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model_gray.names[cls]
            cv2.rectangle(display_gray, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(display_gray, f"{label} {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # Draw results for color model (blue boxes)
    for r in results_style:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model_style.names[cls]
            cv2.rectangle(display_style, (x1, y1), (x2, y2), (255,0,0), 2)
            cv2.putText(display_style, f"{label} {conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    # --- Combine both displays side by side ---
    combined = np.hstack((display_gray, display_style))

    cv2.imshow("Gray YOLO (Left) | Color YOLO (Right)", combined)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
