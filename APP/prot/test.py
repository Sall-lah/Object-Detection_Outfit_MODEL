import onnxruntime as ort
import cv2
import numpy as np

CLASSES = ["Hoodie", "Jacket", "Dress", "Pants", "..."]  # <-- fill yours

def preprocess(img, size=(640,640)):
    img0 = img.copy()
    img = cv2.resize(img, size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0

    # gray model check
    if len(img0.shape) == 2:  # grayscale
        img = np.expand_dims(img, axis=2)

    img = np.transpose(img, (2,0,1))  # HWC->CHW
    img = np.expand_dims(img, 0)
    return img0, img


def xywh_to_xyxy(x):
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # y2
    return y


def nms(boxes, scores, iou=0.5):
    idxs = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.25, nms_threshold=iou)
    return idxs


def postprocess(pred):
    pred = np.squeeze(pred).T     # (8400, 26)

    boxes_xywh = pred[:, :4]      # pixel xywh
    obj = pred[:, 4]
    cls_scores = pred[:, 5:]      # shape (8400, 21)

    # get best class per detection
    class_ids = np.argmax(cls_scores, axis=1)
    scores = cls_scores[np.arange(len(cls_scores)), class_ids]

    # filter low confidence
    mask = scores > 0.01
    boxes_xywh = boxes_xywh[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    # convert xywh -> xyxy (still in pixels)
    xyxy = xywh_to_xyxy(boxes_xywh).astype(int)

    # NMS expects list format
    b = xyxy.tolist()
    s = scores.tolist()

    idxs = cv2.dnn.NMSBoxes(b, s, 0.01, 0.4)

    results = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            results.append({
                "box": b[i],
                "class": int(class_ids[i]),
                "score": float(scores[i])
            })

    return results


# ========================== RUN YOLO ONNX ==========================

session = ort.InferenceSession("utils/best.onnx")
input_name = session.get_inputs()[0].name

img = cv2.imread("img_list/test.jpg")
img0, blob = preprocess(img)

pred = session.run(None, {input_name: blob})[0]
results = postprocess(pred, img0.shape)

# draw
for r in results:
    x1, y1, x2, y2 = r["box"]
    cls = CLASSES[r["class"]]
    score = r["score"]

    cv2.rectangle(img0, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.putText(img0, f"{cls} {score:.2f}", (x1, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

cv2.imwrite("output.jpg", img0)
print("Saved output.jpg")

print("Prediction shape:", pred.shape)
print("First 5 values:", pred.flatten()[:10])