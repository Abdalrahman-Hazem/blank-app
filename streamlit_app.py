import streamlit as st
import numpy as np
import cv2
import onnxruntime as ort

# --- Configuration ---
MODEL_PATH = "best.onnx"
CLASS_NAMES = ['NO mask', 'NOhairnet', 'hairnet', 'mask']
INPUT_SIZE = 640  # Assuming model trained with 640x640 images

# --- Load ONNX model ---
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

# --- Preprocessing ---
def preprocess(image):
    image_resized = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image_chw = np.transpose(image_rgb, (2, 0, 1))  # HWC to CHW
    return np.expand_dims(image_chw, axis=0).astype(np.float32)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def postprocess(preds, input_shape, orig_shape, conf_thresh=0.25):
    boxes, scores, class_ids = [], [], []
    input_h, input_w = input_shape
    orig_h, orig_w = orig_shape

    for pred in preds:
        x, y, w, h = pred[:4]
        obj_conf = sigmoid(pred[4])
        class_confs = sigmoid(pred[5:])
        cls = np.argmax(class_confs)
        cls_conf = class_confs[cls]
        conf = obj_conf * cls_conf

        if conf < conf_thresh:
            continue

        # Box conversion
        x1 = int((x - w / 2) / input_w * orig_w)
        y1 = int((y - h / 2) / input_h * orig_h)
        x2 = int((x + w / 2) / input_w * orig_w)
        y2 = int((y + h / 2) / input_h * orig_h)

        boxes.append([x1, y1, x2, y2])
        scores.append(float(conf))
        class_ids.append(int(cls))

    return boxes, scores, class_ids

# --- Draw Detections ---
def draw_boxes(image, boxes, scores, class_ids):
    from collections import Counter
    counter = Counter()

    for box, score, cls_id in zip(boxes, scores, class_ids):
        if cls_id < 0 or cls_id >= len(CLASS_NAMES):
            continue
        label = f"{CLASS_NAMES[cls_id]}: {score:.2f}"
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, max(y1 - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
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
        st.error("Could not read image.")
    else:
        image_input = preprocess(original_image)
        outputs = session.run(None, {input_name: image_input})
        boxes, scores, class_ids = postprocess(outputs[0], orig_shape=original_image.shape[:2])

        image_with_boxes, counts = draw_boxes(original_image.copy(), boxes, scores, class_ids)
        st.image(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB), caption="Detections", use_container_width=True)

        st.subheader("📊 Class Counts")
        if counts:
            for cls, count in counts.items():
                st.write(f"{cls}: {count}")
        else:
            st.write("No detections found.")