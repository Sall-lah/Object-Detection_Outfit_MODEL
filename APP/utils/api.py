from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("utils/best.pt")  # change to your trained model

def predict(image_path):
    # Read the image
    frame_color = cv2.imread(image_path)

    if frame_color is None:
        raise FileNotFoundError("Image not found!")

    gray_frame = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

    # Run YOLO detection 
    results = model(gray_frame)

    # Find the box with the highest confidence 
    best_box = None
    best_conf = 0
    best_cls = None

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf > best_conf:
                best_conf = conf
                best_box = box
                best_cls = int(box.cls[0])  # class index

    # Crop the detected region 
    if best_box is not None:
        x1, y1, x2, y2 = map(int, best_box.xyxy[0])
        return x1, y1, x2, y2, model.names[best_cls]
    else:
        print("⚠️ No objects detected in the image.")

x1, y1, x2, y2, name = predict("img_list/test.jpg")
print(x1, y1, x2, y2, name)
