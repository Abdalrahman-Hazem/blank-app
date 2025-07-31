import streamlit as st
import numpy as np
import cv2
import onnxruntime as ort
from collections import Counter

# ---- Config ----
MODEL_PATH = "best.onnx"
CLASS_NAMES = ['NO mask', 'NOhairnet', 'hairnet', 'mask']
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
CONFIDENCE_THRESHOLD = 0.25

# ---- Load ONNX Model ----
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name

# ---- Preprocessing ----
def preprocess(image):
    image = cv2.resize(image, (INPUT_WIDTH, INPUT_HEIGHT))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # HWC to CHW
    image = np.expand_dims(image, axis=0)   # Add batch dim
    return image

# ---- Postprocessing ----
def postprocess(outputs):
    boxes, scores, class_ids = [], [], []
    predictions = outputs[0][0]

    for pred in predictions:
        pred = pred[:6]  # keep only first 6 elements
        x1, y1, x2, y2, conf, cls = pred
        if conf < CONFIDENCE_THRESHOLD:
            continue
        boxes.append([int(x1), int(y1), int(x2), int(y2)])
        scores.append(float(conf))
        class_ids.append(int(cls))
    return boxes, scores, class_ids

# ---- Draw Results ----
def draw_boxes(image, boxes, scores, class_ids):
    counter = Counter()
    for box, score, cls_id in zip(boxes, scores, class_ids):
        if cls_id < 0 or cls_id >= len(CLASS_NAMES):
            continue  # Skip unknown classes

        label = f"{CLASS_NAMES[cls_id]}: {score:.2f}"
        color = (0, 255, 0)
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        counter[CLASS_NAMES[cls_id]] += 1
    return image, counter

# ---- Streamlit UI ----
st.set_page_config(page_title="Mask & Hairnet Detection (ONNX)", layout="wide")
st.title("😷 Mask & Hairnet Detection (ONNX Runtime)")

confidence = st.sidebar.slider("Confidence Threshold", 0.25, 1.0, 0.5, 0.05)
CONFIDENCE_THRESHOLD = confidence

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, 1)
    image_for_model = preprocess(original_image)

    outputs = session.run(None, {input_name: image_for_model})
    boxes, scores, class_ids = postprocess(outputs)

    image_with_boxes, counts = draw_boxes(original_image.copy(), boxes, scores, class_ids)
    st.image(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB), channels="RGB", caption="Detections")

    st.sidebar.subheader("📊 Class Counts")
    for cls, count in counts.items():
        st.sidebar.write(f"{cls}: {count}")
    if not counts:
        st.sidebar.write("No detections.")