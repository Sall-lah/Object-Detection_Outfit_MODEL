import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load Class Name
className = ['Hoodie', 'Jacket', 'Mid-lenght dress', 'Pants', 'Shirt', 'coat', 'dress', 'fabric', 'jacket', 'jean', 'm2m', 'plain', 'shirt', 'short', 'shorts', 'skirt', 'slacks', 'suit', 'sweat', 'tie', 'tracksuit', 'tshirt']

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="prot/best_float32.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# Load and preprocess image
img = cv2.imread("img_list/tshirt_gray")

# grayscale → expand to RGB
if len(img.shape) == 2 or img.shape[2] == 1:
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
else:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

orig_h, orig_w = img.shape[:2]

# YOLO input size
img_resized = cv2.resize(img, (640, 640))
img_input = img_resized.astype(np.float32) / 255.0
img_input = np.expand_dims(img_input, axis=0)   # (1,640,640,3)

# Feed model input
interpreter.set_tensor(input_details[0]['index'], img_input)
interpreter.invoke()

# Get model output (1, 26, 8400)
output = interpreter.get_tensor(output_details[0]['index'])[0]   # (26,8400)


# Exctract YOLO Output

# split into bbox + class logits
bboxes = output[0:4, :]         # xywh, shape (4,8400)
class_logits = output[4:, :]    # (22 classes), shape (22,8400)

# apply sigmoid
bboxes = 1 / (1 + np.exp(-bboxes))
class_probs = 1 / (1 + np.exp(-class_logits))

confidences = np.max(class_probs, axis=0)
class_ids   = np.argmax(class_probs, axis=0)

# threshold
mask = confidences > 0.25
confidences = confidences[mask]
class_ids   = class_ids[mask]
bboxes      = bboxes[:, mask]


# Convert xywh → x1,y1,x2,y2
x, y, w, h = bboxes

x1 = (x - w/2) * orig_w
y1 = (y - h/2) * orig_h
x2 = (x + w/2) * orig_w
y2 = (y + h/2) * orig_h

boxes = np.vstack([y1, x1, y2, x2]).T.astype(np.float32)


# Apply Non-Max Suppression
selected = tf.image.non_max_suppression(
    boxes,
    confidences.astype(np.float32),
    max_output_size=100,
    iou_threshold=0.45,
    score_threshold=0.25
).numpy()


# Draw boxes and plot

img_plot = img.copy()

for i in selected:
    yy1, xx1, yy2, xx2 = boxes[i].astype(int)
    cls = className[int(class_ids[i])]
    conf = float(confidences[i])

    cv2.rectangle(img_plot, (xx1, yy1), (xx2, yy2), (255, 0, 0), 2)
    cv2.putText(img_plot, f"{cls} {conf:.2f}",
                (xx1, yy1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 0, 0), 2)


plt.figure(figsize=(8, 8))
plt.imshow(img_plot)
plt.axis("off")
plt.show()