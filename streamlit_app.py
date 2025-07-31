import streamlit as st
import numpy as np
import cv2
import onnxruntime as ort

# --- Config ---
MODEL_PATH = "best.onnx"
CLASS_NAMES = ['NO mask', 'NOhairnet', 'hairnet', 'mask', 'mask+hairnet']  # Assume 5 classes just in case
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
CONFIDENCE_THRESHOLD = 0.01  # DEBUG: very low to test detections

# --- Load ONNX model ---
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name

# --- Preprocessing ---
def preprocess(image):
    image = cv2.resize(image, (INPUT_WIDTH, INPUT_HEIGHT))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image

# --- Postprocessing ---
def postprocess(preds, input_shape, orig_shape, conf_thresh):
    boxes, scores, class_ids = [], [], []
    input_h, input_w = input_shape
    orig_h, orig_w = orig_shape

    for pred in preds:
        if len(pred) < 6:
            continue

        x, y, w, h, obj_conf = pred[:5]
        class_confs = pred[5:]
        cls = np.argmax(class_confs)
        conf = obj_conf * class_confs[cls]

        if conf < conf_thresh:
            continue

        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2

        # Rescale to original image size
        x1 = int(x1 / input_w * orig_w)
        y1 = int(y1 / input_h * orig_h)
        x2 = int(x2 / input_w * orig_w)
        y2 = int(y2 / input_h * orig_h)

        boxes.append([x1, y1, x2, y2])
        scores.append(float(conf))
        class_ids.append(int(cls))

    return boxes, scores, class_ids

# --- Draw boxes ---
def draw_boxes(image, boxes, scores, class_ids):
    for box, score, cls_id in zip(boxes, scores, class_ids):
        color = (0, 255, 0)
        x1, y1, x2, y2 = box
        label = f"{cls_id}: {score:.2f}"
        if 0 <= cls_id < len(CLASS_NAMES):
            label = f"{CLASS_NAMES[cls_id]}: {score:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

# --- Streamlit UI ---
st.set_page_config(page_title="YOLOv8 ONNX Debug", layout="wide")
st.title("🧪 YOLOv8 ONNX Detection Debugger")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, 1)
    input_shape = (INPUT_HEIGHT, INPUT_WIDTH)
    orig_shape = original_image.shape[:2]

    image_for_model = preprocess(original_image)
    outputs = session.run(None, {input_name: image_for_model})
    preds = np.squeeze(outputs[0]).T  # (8400, 8)

    st.write("Sample Output Row:", preds[0])
    st.write("Output Shape:", preds.shape)

    boxes, scores, class_ids = postprocess(preds, input_shape, orig_shape, CONFIDENCE_THRESHOLD)

    st.write(f"Detections Found: {len(boxes)}")
    if len(boxes) > 0:
        image_with_boxes = draw_boxes(original_image.copy(), boxes, scores, class_ids)
        st.image(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB), caption="Detections", channels="RGB", use_container_width=True)
    else:
        st.warning("No detections found above the confidence threshold.")