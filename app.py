from ultralytics import YOLO

# Load pretrained model (e.g. YOLOv8s)
model = YOLO("yolov8s.pt")

# Run detection on an image or video
results = model("image.jpg", show=True)

# Print results
for r in results:
    print(r.boxes.xyxy)  # bounding box coordinates
