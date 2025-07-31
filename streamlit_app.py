import streamlit as st
import numpy as np
import cv2
import onnxruntime as ort
from collections import Counter

# --- Configuration ---
MODEL_PATH = "best.onnx"
CLASS_NAMES = ['NO mask', 'NOhairnet', 'hairnet', 'mask']
INPUT_SIZE = 640  # Width and height

# --- Load ONNX model ---
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

# --- Utils ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def preprocess(image):
    image_resized = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image_chw = np.transpose(image_rgb, (2, 0, 1))  # HWC to CHW
    return np.expand_dims(image_chw, axis=0).astype(np.float32)

def postprocess(preds, input_shape, orig_shape, conf_thresh=0.25):
    boxes, scores, class_ids = [], [], []
    input_h, input_w = input_shape
    orig_h, orig_w = orig_shape

    for i, pred in enumerate(preds):
        if len(pred) < 6:
            continue

        # YOLOv8 format: x_center, y_center, width, height, obj_conf, cls1_conf, cls2_conf, ...
        x, y, w, h = pred[:4]
        obj_conf = sigmoid(pred[4])
        class_probs = sigmoid(pred[5:])
        cls_id = np.argmax(class_probs)
        cls_conf = class_probs[cls_id]

        conf = obj_conf * cls_conf
        if conf < conf_thresh:
            continue

        # Get box coordinates in the model's scale (0-640)
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2

        # Scale boxes to original image shape
        x1 = int(x1 / input_w * orig_w)
        y1 = int(y1 / input_h * orig_h)
        x2 = int(x2 / input_w * orig_w)
        y2 = int(y2 / input_h * orig_h)

        # Clamp box coordinates
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, orig_w), min(y2, orig_h)

        boxes.append([x1, y1, x2, y2])
        scores.append(float(conf))
        class_ids.append(int(cls_id))

    return boxes, scores, class_ids

def draw_boxes(image, boxes, scores, class_ids):
    counter = Counter()
    for box, score, cls_id in zip(boxes, scores, class_ids):
        if 0 <= cls_id < len(CLASS_NAMES):
            label = f"{CLASS_NAMES[cls_id]}: {score:.2f}"
            x1, y1, x2, y2 = box
            color = (0, 255, 0)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, max(y1 - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            counter[CLASS_NAMES[cls_id]] += 1
    return image, counter

# --- Streamlit UI ---
st.set_page_config(page_title="Mask & Hairnet Detection", layout="wide")
st.title("😷 Mask & Hairnet Detection (YOLOv8 ONNX)")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, 1)
    if original_image is None:
        st.error("❌ Could not read the image. Please upload a valid image.")
    else:
        orig_shape = original_image.shape[:2]  # (h, w)
        input_shape = (INPUT_SIZE, INPUT_SIZE)
        image_input = preprocess(original_image)

        outputs = session.run(None, {input_name: image_input})
        preds = np.squeeze(outputs[0])  # Shape: (8400, 8)
        st.write("Sample confidences:", preds[:5, 5:])
        boxes, scores, class_ids = postprocess(preds, input_shape, orig_shape)

        image_with_boxes, counts = draw_boxes(original_image.copy(), boxes, scores, class_ids)
        st.image(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB), caption="🖼 Detections", use_container_width=True)

        st.subheader("📊 Class Counts")
        if counts:
            for cls, count in counts.items():
                st.write(f"{cls}: {count}")
        else:
            st.warning("No detections found.")