from ultralytics import YOLO
import cv2
import numpy as np
from sklearn.cluster import KMeans
import webcolors
import matplotlib.pyplot as plt

# Helper: find nearest color name
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


# ---------- Step 1: Load YOLO model ----------
model = YOLO("final/final3_gray/weights/best.pt")  # change to your trained model

# ---------- Step 2: Read the image ----------
image_path = "test/handsome-casual-man-standing-arms-260nw-485380684.webp"  # change to your image
frame = cv2.imread(image_path)
if frame is None:
    raise FileNotFoundError("Image not found!")

# ---------- Step 3: Run YOLO detection ----------
results = model(frame)

# ---------- Step 4: Find the box with the highest confidence ----------
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

# ---------- Step 5: Crop the detected region ----------
if best_box is not None:
    x1, y1, x2, y2 = map(int, best_box.xyxy[0])
    cropped = frame[y1:y2, x1:x2]

    # Convert to RGB for KMeans
    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

    # ---------- Step 6: Run K-Means ----------
    pixels = cropped_rgb.reshape(-1, 3)
    kmeans = KMeans(n_clusters=3, random_state=0)
    kmeans.fit(pixels)

    colors = np.array(kmeans.cluster_centers_, dtype='uint8')
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    dominant_color = colors[np.argmax(counts)]

    # ---------- Step 7: Get color name ----------
    color_name = rgb_to_name(dominant_color)
    print(f"Most Dominant Color (RGB): {dominant_color}")
    print(f"Color Name: {color_name}")

    # ---------- Step 8: Display ----------
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    plt.title(f"Detected Object: {model.names[best_cls]}")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(np.ones((100,100,3), dtype='uint8') * dominant_color)
    plt.title(f"Dominant Color: {color_name}")
    plt.axis("off")

    plt.show()
else:
    print("⚠️ No objects detected in the image.")
