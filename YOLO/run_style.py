from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO("final/final4_style/weights/best.pt")  # or yolov8n_gray.pt

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

# Optional: set resolution
cap.set(3, 640)
cap.set(4, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference
    results = model(frame, stream=True)

    # Loop over results and draw boxes
    for r in results:
        for box in r.boxes:
            # Get coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Get class name
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            name = model.names[cls]

            # Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show on screen
    cv2.imshow("Outfit-Invetory-Detection-Model", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()