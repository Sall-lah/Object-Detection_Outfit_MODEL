from ultralytics import YOLO

model = YOLO("utils/best.pt")   # or best.pt
model.export(format="tflite")