import streamlit as st
import numpy as np
import cv2
import onnxruntime as ort
from collections import Counter

# --- Config ---
MODEL_PATH = "best.onnx"
CLASS_NAMES = ['NO mask', 'NOhairnet', 'hairnet', 'mask']
INPUT_WIDTH = 640
INPUT_HEIGHT = 640

# --- Load ONNX model ---
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name

# --- Preprocessing ---
def preprocess(image):
    image = cv2.resize(image, (INPUT_WIDTH, INPUT_HEIGHT))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # HWC to CHW
    image = np.expand_dims(image, axis=0)
    return image

# --- Postprocessing ---
def postprocess(preds, input_shape, orig_shape, conf_thresh=0.25):
    boxes, scores, class_ids = [], [], []
    input_h, input_w = input_shape
    orig_h, orig_w = orig_shape

    for pred in preds:
        if len(pred) != 8:
            continue

        x, y, w, h, obj_conf, c0, c1, c2 = pred
        class_confs = [c0, c1, c2]
        cls_id = int(np.argmax(class_confs))
        cls_conf = class_confs[cls_id]
        conf = obj_conf * cls_conf

        if conf < conf_thresh:
            continue

        # Convert center xywh to x1y1x2y2
        x1 = int((x - w / 2) / input_w * orig_w)
        y1 = int((y - h / 2) / input_h * orig_h)
        x2 = int((x + w / 2) / input_w * orig_w)
        y2 = int((y + h / 2) / input_h * orig_h)

        boxes.append([x1, y1, x2, y2])
        scores.append(float(conf))
        class_ids.append(cls_id)

    return boxes, scores, class_ids

# --- Draw bounding boxes ---
def draw_boxes(image, boxes, scores, class_ids):
    counter = Counter()
    for box, score, cls_id in zip(boxes, scores, class_ids):
        if cls_id < 0 or cls_id >= len(CLASS_NAMES):
            continue
        label = f"{CLASS_NAMES[cls_id]}: {score:.2f}"
        color = (0, 255, 0)
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        counter[CLASS_NAMES[cls_id]] += 1
    return image, counter

# --- Streamlit UI ---
st.set_page_config(page_title="Mask & Hairnet Detection", layout="wide")
st.title("😷 Mask & Hairnet Detection (YOLOv8 ONNX)")

confidence = st.sidebar.slider("Confidence Threshold", 0.2, 1.0, 0.3, 0.05)

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, 1)

    input_shape = (INPUT_HEIGHT, INPUT_WIDTH)
    orig_shape = original_image.shape[:2]
    image_for_model = preprocess(original_image)

    outputs = session.run(None, {input_name: image_for_model})
    raw_output = outputs[0]              # (1, 8, 8400)
    preds = np.squeeze(raw_output).T     # (8400, 8)

    boxes, scores, class_ids = postprocess(preds, input_shape, orig_shape, confidence)

    image_with_boxes, counts = draw_boxes(original_image.copy(), boxes, scores, class_ids)
    st.image(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB), caption="Detections", channels="RGB", use_container_width=True)

    st.sidebar.subheader("📊 Class Counts")
    if counts:
        for cls, count in counts.items():
            st.sidebar.write(f"{cls}: {count}")
    else:
        st.sidebar.write("No detections found.")