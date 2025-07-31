import cv2
import numpy as np
import streamlit as st
import onnxruntime as ort
from PIL import Image

# Config
CLASS_NAMES = ['NO mask', 'NOhairnet', 'hairnet', 'mask']
CONFIDENCE_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45
INPUT_SIZE = (640, 640)  # width, height

def preprocess(image):
    resized = cv2.resize(image, INPUT_SIZE)
    normalized = resized.astype(np.float32) / 255.0
    normalized = normalized.transpose(2, 0, 1)  # HWC -> CHW
    return normalized[np.newaxis, :, :, :]  # Add batch dimension

def xywh2xyxy(box):
    """Convert YOLO (x_center, y_center, w, h) to (x1, y1, x2, y2)"""
    x_c, y_c, w, h = box
    return [x_c - w / 2, y_c - h / 2, x_c + w / 2, y_c + h / 2]

def iou(box1, box2):
    """Compute IoU between two boxes"""
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    inter_x1 = max(x1, x1g)
    inter_y1 = max(y1, y1g)
    inter_x2 = min(x2, x2g)
    inter_y2 = min(y2, y2g)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    area1 = max(0, x2 - x1) * max(0, y2 - y1)
    area2 = max(0, x2g - x1g) * max(0, y2g - y1g)

    union_area = area1 + area2 - inter_area + 1e-6
    return inter_area / union_area

def non_max_suppression(boxes, scores, threshold):
    """Pure NumPy NMS"""
    indices = np.argsort(scores)[::-1]
    keep = []

    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        remaining = indices[1:]

        ious = np.array([iou(boxes[current], boxes[i]) for i in remaining])
        indices = remaining[ious < threshold]

    return keep

def postprocess(preds, original_shape):
    preds = preds.squeeze().T  # (8400, 8)
    boxes, scores, class_ids = [], [], []

    for pred in preds:
        x_c, y_c, w, h = pred[:4]
        cls_scores = pred[4:]
        class_id = int(np.argmax(cls_scores))
        confidence = cls_scores[class_id]

        if confidence > CONFIDENCE_THRESHOLD and class_id < len(CLASS_NAMES):
            box = xywh2xyxy([x_c, y_c, w, h])
            boxes.append(box)
            scores.append(confidence)
            class_ids.append(class_id)

    if not boxes:
        return [], [], []

    boxes = np.array(boxes)
    scores = np.array(scores)
    keep_indices = non_max_suppression(boxes, scores, NMS_THRESHOLD)

    final_boxes, final_scores, final_classes = [], [], []
    scale_x = original_shape[1] / INPUT_SIZE[0]
    scale_y = original_shape[0] / INPUT_SIZE[1]

    for i in keep_indices:
        x1, y1, x2, y2 = boxes[i]
        final_boxes.append([
            int(x1 * scale_x),
            int(y1 * scale_y),
            int(x2 * scale_x),
            int(y2 * scale_y)
        ])
        final_scores.append(float(scores[i]))
        final_classes.append(class_ids[i])

    return final_boxes, final_scores, final_classes

def draw_boxes(image, boxes, scores, class_ids):
    counts = {cls: 0 for cls in CLASS_NAMES}
    for box, score, cls_id in zip(boxes, scores, class_ids):
        label = f"{CLASS_NAMES[cls_id]}: {score:.2f}"
        color = (0, 255, 0)
        cv2.rectangle(image, box[:2], box[2:], color, 2)
        cv2.putText(image, label, (box[0], max(box[1] - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        counts[CLASS_NAMES[cls_id]] += 1
    return image, counts

# 🔘 Streamlit UI
st.title("🛡️ Mask & Hairnet Detection (YOLOv8 ONNX)")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # Load ONNX model
    session = ort.InferenceSession("best.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # Run inference
    img_input = preprocess(image_np)
    preds = session.run(None, {input_name: img_input})[0]

    # Postprocess predictions
    boxes, scores, class_ids = postprocess(preds, image_np.shape[:2])
    result_img, counts = draw_boxes(image_np.copy(), boxes, scores, class_ids)

    # Display result
    st.image(result_img, caption="🎯 Detection Result", use_container_width=True)

    st.subheader("📊 Class Counts")
    for cls, count in counts.items():
        if count > 0:
            st.write(f"**{cls}**: {count}")