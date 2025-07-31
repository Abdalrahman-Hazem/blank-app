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
CONFIDENCE_THRESHOLD = 0.3

# --- Load ONNX Model ---
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

# --- Preprocess ---
def preprocess(image):
    resized = cv2.resize(image, (INPUT_WIDTH, INPUT_HEIGHT))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    transposed = np.transpose(normalized, (2, 0, 1))
    expanded = np.expand_dims(transposed, axis=0)
    return expanded, resized.shape[:2]  # (input for model, original resized shape)

# --- Postprocess ---
def postprocess(predictions, input_shape, conf_thresh=CONFIDENCE_THRESHOLD):
    boxes, scores, class_ids = [], [], []
    input_h, input_w = input_shape
    output = predictions[0].squeeze().T  # shape: (8400, 8)

    for pred in output:
        x, y, w, h, obj_conf, *class_probs = pred
        cls = np.argmax(class_probs)
        cls_conf = class_probs[cls]
        conf = obj_conf * cls_conf

        if conf < conf_thresh:
            continue

        x1 = int((x - w / 2) / input_w * INPUT_WIDTH)
        y1 = int((y - h / 2) / input_h * INPUT_HEIGHT)
        x2 = int((x + w / 2) / input_w * INPUT_WIDTH)
        y2 = int((y + h / 2) / input_h * INPUT_HEIGHT)

        boxes.append([x1, y1, x2, y2])
        scores.append(float(conf))
        class_ids.append(int(cls))

    return boxes, scores, class_ids

# --- Draw ---
def draw_boxes(image, boxes, scores, class_ids):
    counter = Counter()
    for box, score, cls_id in zip(boxes, scores, class_ids):
        if cls_id < 0 or cls_id >= len(CLASS_NAMES):
            continue
        label = f"{CLASS_NAMES[cls_id]}: {score:.2f}"
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        counter[CLASS_NAMES[cls_id]] += 1
    return image, counter

# --- Streamlit UI ---
st.set_page_config(page_title="Mask & Hairnet Detection", layout="wide")
st.title("😷 Mask & Hairnet Detection (YOLOv8 ONNX)")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    input_tensor, input_shape = preprocess(image)
    predictions = session.run(None, {input_name: input_tensor})
    boxes, scores, class_ids = postprocess(predictions, input_shape)

    result_img, counts = draw_boxes(image.copy(), boxes, scores, class_ids)
    st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), channels="RGB", caption="Detections", use_container_width=True)

    st.sidebar.subheader("📊 Class Counts")
    if counts:
        for cls, count in counts.items():
            st.sidebar.write(f"{cls}: {count}")
    else:
        st.sidebar.write("No detections found.")