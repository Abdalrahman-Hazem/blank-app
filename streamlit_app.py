import cv2
import numpy as np
import streamlit as st
import onnxruntime as ort
from PIL import Image, ImageOps

# Configuration
CLASS_NAMES = ['NO mask', 'NOhairnet', 'hairnet', 'mask']
CONFIDENCE_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45
INPUT_SIZE = (640, 640)

# Preprocess image
def preprocess(image_pil):
    image_pil = ImageOps.exif_transpose(image_pil)
    image = np.array(image_pil)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    resized = cv2.resize(image, INPUT_SIZE)
    input_tensor = resized.transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32) / 255.0
    return input_tensor, image

# Postprocess predictions
def postprocess(preds, original_shape):
    preds = preds.squeeze().T  # (8400, 8)
    boxes, confidences, class_ids = [], [], []

    for pred in preds:
        x1, y1, x2, y2 = pred[:4]
        cls_scores = pred[4:]
        class_id = int(np.argmax(cls_scores))
        confidence = float(cls_scores[class_id])

        if confidence > CONFIDENCE_THRESHOLD and class_id < len(CLASS_NAMES):
            boxes.append([x1, y1, x2, y2])
            confidences.append(confidence)
            class_ids.append(class_id)

    final_boxes, final_scores, final_classes = [], [], []
    if boxes:
        boxes_np = np.array(boxes)
        scores_np = np.array(confidences)

        # Convert boxes to [x, y, w, h] for NMS
        boxes_xywh = np.array([[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in boxes_np])
        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes_xywh.tolist(),
            scores=scores_np.tolist(),
            score_threshold=CONFIDENCE_THRESHOLD,
            nms_threshold=NMS_THRESHOLD
        )

        scale_x = original_shape[1] / INPUT_SIZE[0]
        scale_y = original_shape[0] / INPUT_SIZE[1]

        for i in indices.flatten():
            x1, y1, x2, y2 = boxes_np[i]
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            final_boxes.append([x1, y1, x2, y2])
            final_scores.append(confidences[i])
            final_classes.append(class_ids[i])

    return final_boxes, final_scores, final_classes

# Draw bounding boxes
def draw_boxes(image, boxes, scores, class_ids):
    counts = {cls: 0 for cls in CLASS_NAMES}
    for box, score, cls_id in zip(boxes, scores, class_ids):
        label = f"{CLASS_NAMES[cls_id]}: {score:.2f}"
        color = (0, 255, 0)
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)
        cv2.putText(image, label, (box[0], max(box[1] - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        counts[CLASS_NAMES[cls_id]] += 1
    return image, counts

# Streamlit App
st.set_page_config(page_title="Mask & Hairnet Detection", layout="centered")
st.title("😷 Mask & Hairnet Detection (YOLOv8 + ONNX)")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image_pil = Image.open(uploaded_file).convert("RGB")
    input_tensor, image_bgr = preprocess(image_pil)
    original_shape = image_bgr.shape[:2]

    # Load ONNX model
    session = ort.InferenceSession("best.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # Inference
    preds = session.run(None, {input_name: input_tensor})[0]

    # Postprocessing
    boxes, scores, class_ids = postprocess(preds, original_shape)
    result_img, counts = draw_boxes(image_bgr.copy(), boxes, scores, class_ids)

    # Show results
    st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="🎯 Detection Result", use_container_width=True)

    # Display class counts
    st.subheader("📊 Class Counts")
    for cls, count in counts.items():
        if count > 0:
            st.write(f"- **{cls}**: {count}")