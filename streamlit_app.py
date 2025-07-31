import streamlit as st
import numpy as np
import cv2
import onnxruntime as ort
from collections import Counter

# --- Config ---
MODEL_PATH = "best.onnx"
CLASS_NAMES = ['NO mask', 'NOhairnet', 'hairnet', 'mask']
INPUT_SIZE = 640  # YOLOv8 default

# --- Load model ---
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name

# --- Preprocessing ---
def preprocess(image):
    img = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]  # (1, 3, H, W)
    return img

# --- Postprocessing ---
def postprocess(predictions, conf_thresh=0.25):
    boxes, scores, class_ids = [], [], []
    preds = predictions[0].T  # (8400, 8)
    for pred in preds:
        if len(pred) < 6:
            continue
        x, y, w, h, obj_conf = pred[:5]
        class_probs = pred[5:]
        cls = np.argmax(class_probs)
        conf = obj_conf * class_probs[cls]
        if conf < conf_thresh:
            continue
        x1 = int((x - w / 2) / INPUT_SIZE * orig_w)
        y1 = int((y - h / 2) / INPUT_SIZE * orig_h)
        x2 = int((x + w / 2) / INPUT_SIZE * orig_w)
        y2 = int((y + h / 2) / INPUT_SIZE * orig_h)
        boxes.append([x1, y1, x2, y2])
        scores.append(float(conf))
        class_ids.append(int(cls))
    return boxes, scores, class_ids

# --- Drawing ---
def draw_boxes(image, boxes, scores, class_ids):
    count = Counter()
    for box, score, cls_id in zip(boxes, scores, class_ids):
        label = f"{CLASS_NAMES[cls_id]} {score:.2f}" if cls_id < len(CLASS_NAMES) else f"Class {cls_id}"
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, max(y1 - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        count[CLASS_NAMES[cls_id]] += 1
    return image, count

# --- Streamlit UI ---
st.set_page_config(page_title="Mask & Hairnet Detection", layout="wide")
st.title("😷 Mask & Hairnet Detection (YOLOv8 ONNX)")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, 1)
    orig_h, orig_w = original_image.shape[:2]

    input_image = preprocess(original_image)
    outputs = session.run(None, {input_name: input_image})
    boxes, scores, class_ids = postprocess(outputs)

    image_with_boxes, counts = draw_boxes(original_image.copy(), boxes, scores, class_ids)
    st.image(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB), caption="Detections", use_container_width=True)

    st.sidebar.header("📊 Detection Counts")
    if counts:
        for cls, cnt in counts.items():
            st.sidebar.write(f"{cls}: {cnt}")
    else:
        st.sidebar.write("No detections found.")