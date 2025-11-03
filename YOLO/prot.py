from ultralytics import YOLO
import cv2
import numpy as np
from sklearn.cluster import KMeans
import webcolors
import matplotlib.pyplot as plt

# ---------- Function: map RGB → closest color name ----------
def rgb_to_name(rgb_color):
    import webcolors
    from collections import namedtuple

    # Convert BGR → RGB
    r, g, b = int(rgb_color[2]), int(rgb_color[1]), int(rgb_color[0])

    # Try exact color name first
    try:
        return webcolors.rgb_to_name((r, g, b))
    except ValueError:
        # Build fallback color dictionary (covers all versions)
        try:
            color_dict = webcolors.CSS3_NAMES_TO_HEX
        except AttributeError:
            # Older or newer versions expose names differently
            try:
                color_dict = webcolors._definitions._CSS3_NAMES_TO_HEX
            except Exception:
                # Manual minimal fallback
                color_dict = {
                    'black': '#000000', 'white': '#ffffff', 'red': '#ff0000',
                    'green': '#008000', 'blue': '#0000ff', 'yellow': '#ffff00',
                    'cyan': '#00ffff', 'magenta': '#ff00ff', 'gray': '#808080'
                }

        # Find the closest match
        min_diff = float('inf')
        closest_name = "unknown"
        for name, hex_val in color_dict.items():
            rc, gc, bc = webcolors.hex_to_rgb(hex_val)
            diff = (r - rc) ** 2 + (g - gc) ** 2 + (b - bc) ** 2
            if diff < min_diff:
                min_diff = diff
                closest_name = name

        return closest_name


# ---------- Step 1 – Load models and image ----------
gray_model = YOLO("final/final3_gray/weights/best.pt")   # grayscale-trained model
image_path = "test/image.png"
frame_color = cv2.imread(image_path)
if frame_color is None:
    raise FileNotFoundError("Image not found!")

# Create grayscale version for YOLO detection
frame_gray = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)
frame_gray_3ch = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)  # YOLO expects 3-channel input

# ---------- Step 2 – Run YOLO on grayscale image ----------
results = gray_model(frame_gray_3ch)

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

# ---------- Step 3 – For each detection, crop color region & get K-Means ----------
if best_box is not None:
    conf = float(best_box.conf[0])
    cls  = int(best_box.cls[0])
    name = gray_model.names[cls]

    x1, y1, x2, y2 = map(int, best_box.xyxy[0])
    crop_color = frame_color[y1:y2, x1:x2]


    # # Convert to RGB for K-Means
    crop_rgb = cv2.cvtColor(crop_color, cv2.COLOR_BGR2RGB)

    # ---- K-Means clustering ----
    pixels = crop_rgb.reshape(-1, 3)
    kmeans = KMeans(n_clusters=3, random_state=0)
    kmeans.fit(pixels)

    colors = np.array(kmeans.cluster_centers_, dtype="uint8")
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    dominant_color = colors[np.argmax(counts)]
    color_name = rgb_to_name(dominant_color)

    print(f"Object: {name}, Confidence: {conf:.2f}, Color: {color_name}, RGB: {dominant_color}")

    # ---- Draw results on original image ----
    cv2.rectangle(frame_color, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame_color, f"{name} ({color_name})", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# ---------- Step 4 – Display ----------
cv2.imshow("Detected Objects + Colors", frame_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
