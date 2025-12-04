from ultralytics import YOLO
model = YOLO("yolo11n.pt")
results = model.train(data="datasets/grayscale2/data.yaml", epochs=50, imgsz=640)