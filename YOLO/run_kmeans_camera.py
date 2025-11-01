from ultralytics import YOLO
import cv2
import numpy as np
from sklearn.cluster import KMeans
import webcolors

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

# Load trained YOLO model
model = YOLO("final/final3_gray/weights/best.pt")

# Start camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model(frame, stream=True)

    highest_conf = 0
    best_box = None
    best_cls = None

    # Find detection with highest confidence
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf > highest_conf:
                highest_conf = conf
                best_box = box
                best_cls = int(box.cls[0])

    # If found a valid detection
    if best_box is not None:
        x1, y1, x2, y2 = map(int, best_box.xyxy[0])
        label = model.names[best_cls]
        cropped = frame[y1:y2, x1:x2]

        if cropped.size != 0:
            # Prepare pixels for KMeans
            pixels = cropped.reshape(-1, 3)
            kmeans = KMeans(n_clusters=1, n_init=10)
            kmeans.fit(pixels)
            dominant_color = kmeans.cluster_centers_[0].astype(int)

            # Convert to color name
            color_name = rgb_to_name(dominant_color)

            # Draw detection box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {highest_conf:.2f} {color_name} {dominant_color}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Draw color patch
            color_patch_x1, color_patch_y1 = x1, y2 + 10
            color_patch_x2, color_patch_y2 = x1 + 50, y2 + 60
            cv2.rectangle(frame, (color_patch_x1, color_patch_y1),
                          (color_patch_x2, color_patch_y2),
                          dominant_color.tolist(), -1)

            # Show color name
            cv2.putText(frame, f"Color: {color_name}", (x1, y2 + 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Display
    cv2.imshow("YOLO + Dominant Color Name", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
