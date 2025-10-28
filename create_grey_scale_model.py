from ultralytics import YOLO
import torch
import torch.nn as nn

# Load base YOLOv8 model
model = YOLO("yolov8s.pt")

# Access first Conv block
first_conv = model.model.model[0].conv

# Create a new Conv2d with 1 input channel
new_conv = nn.Conv2d(1, first_conv.out_channels,
                     kernel_size=first_conv.kernel_size,
                     stride=first_conv.stride,
                     padding=first_conv.padding,
                     bias=first_conv.bias is not None)

# Copy pretrained weights (average over RGB channels)
with torch.no_grad():
    new_conv.weight[:] = first_conv.weight.mean(1, keepdim=True)

# Replace original conv layer
model.model.model[0].conv = new_conv

# Save the modified model
model.save("yolov8s_gray.pt")

print("✅ YOLOv8 grayscale model created and saved as yolov8s_gray.pt")