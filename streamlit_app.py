import cv2
import numpy as np
import streamlit as st
import onnxruntime as ort
from PIL import Image

CLASS_NAMES = ['NO mask', 'NOhairnet', 'hairnet', 'mask']
CONFIDENCE_THRESHOLD = 0.25
input_shape = (640, 640)

def preprocess(image):
    image = cv2.resize(image, input_shape)
    image = image.transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32) / 255.0
    return image

def postprocess(preds, orig_shape, input_shape):
    preds = preds.squeeze().T  # (8400, 8)
    boxes, scores, class_ids = [], [], []
    for pred in preds:
        x1, y1, x2, y2 = pred[:4]
        cls_scores = pred[4:]
        class_id = int(np.argmax(cls_scores))
        confidence = cls_scores[class_id]
        if confidence > CONFIDENCE_THRESHOLD and class_id < len(CLASS_NAMES):
            scale_x, scale_y = orig_shape[1] / input_shape[1], orig_shape[0] / input_shape[0]
            boxes.append([int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)])
            scores.append(float(confidence))
            class_ids.append(class_id)
    return boxes, scores, class_ids

def draw_boxes(image, boxes, scores, class_ids):
    counts = {cls: 0 for cls in CLASS_NAMES}
    for box, score, cls_id in zip(boxes, scores, class_ids):
        label = f"{CLASS_NAMES[cls_id]}: {score:.2f}"
        color = (0, 255, 0)
        cv2.rectangle(image, box[:2], box[2:], color, 2)
        cv2.putText(image, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        counts[CLASS_NAMES[cls_id]] += 1
    return image, counts

# ✅ Streamlit UI
st.title("Mask & Hairnet Detection (YOLOv8-ONNX)")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # ONNX Inference
    session = ort.InferenceSession("best.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    img_input = preprocess(image_np)
    preds = session.run(None, {input_name: img_input})[0]

    boxes, scores, class_ids = postprocess(preds, image_np.shape[:2], input_shape)
    result_img, counts = draw_boxes(image_np.copy(), boxes, scores, class_ids)

    st.image(result_img, caption="Detected", use_container_width=True)
    st.subheader("📊 Class Counts")
    for cls, count in counts.items():
        if count > 0:
            st.write(f"**{cls}**: {count}")