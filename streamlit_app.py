import cv2
import numpy as np
import streamlit as st
import onnxruntime as ort
from PIL import Image, ImageOps

# Constants
CLASS_NAMES = ['NO mask', 'NOhairnet', 'hairnet', 'mask']
CONFIDENCE_THRESHOLD = 0.25
INPUT_SIZE = (640, 640)

# Preprocess: PIL → BGR + Resize + Normalize
def preprocess(image_pil):
    image_pil = ImageOps.exif_transpose(image_pil)  # Fix orientation if needed
    image = np.array(image_pil)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    resized = cv2.resize(image, INPUT_SIZE)
    input_tensor = resized.transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32) / 255.0
    return input_tensor, image  # original BGR image for drawing

# Postprocess predictions
def postprocess(preds, orig_shape, input_shape):
NMS_THRESHOLD = 0.45
INPUT_SIZE = (640, 640)  # (width, height)

def preprocess(image):
    resized = cv2.resize(image, INPUT_SIZE)
    normalized = resized.transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32) / 255.0
    return normalized

def postprocess(preds, original_shape):
    preds = preds.squeeze().T  # (8400, 8)
    boxes, scores, class_ids = [], [], []
    for pred in preds:
        x1, y1, x2, y2 = pred[:4]
        cls_scores = pred[4:]
        class_id = int(np.argmax(cls_scores))
        confidence = cls_scores[class_id]
        if confidence > CONFIDENCE_THRESHOLD and class_id < len(CLASS_NAMES):
            # Scale boxes back to original image size
            scale_x, scale_y = orig_shape[1] / input_shape[0], orig_shape[0] / input_shape[1]
            boxes.append([
                int(x1 * scale_x),
                int(y1 * scale_y),
                int(x2 * scale_x),
                int(y2 * scale_y)
            ])
            scores.append(float(confidence))
            boxes.append([x1, y1, x2, y2])  # unscaled
            confidences.append(float(confidence))
            class_ids.append(class_id)
    return boxes, scores, class_ids

# Draw bounding boxes and labels

    # Apply NMS
    final_boxes, final_scores, final_classes = [], [], []
    if len(boxes):
        boxes_np = np.array(boxes)
        scores_np = np.array(confidences)
        indices = cv2.dnn.NMSBoxes(
            bboxes=[[float(x1), float(y1), float(x2 - x1), float(y2 - y1)] for x1, y1, x2, y2 in boxes_np],
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

def draw_boxes(image, boxes, scores, class_ids):
    counts = {cls: 0 for cls in CLASS_NAMES}
    for box, score, cls_id in zip(boxes, scores, class_ids):
        label = f"{CLASS_NAMES[cls_id]}: {score:.2f}"
        color = (0, 255, 0)
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)
        cv2.putText(image, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.rectangle(image, box[:2], box[2:], color, 2)
        cv2.putText(image, label, (box[0], max(box[1] - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        counts[CLASS_NAMES[cls_id]] += 1
    return image, counts

# Streamlit UI
st.set_page_config(page_title="Mask & Hairnet Detection", layout="centered")
st.title("🧠 Mask & Hairnet Detection (YOLOv8 ONNX)")

# 🔘 Streamlit App
st.title("🛡️ Mask & Hairnet Detection (YOLOv8 ONNX)")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load and preprocess
    image_pil = Image.open(uploaded_file).convert("RGB")
    input_tensor, original_image = preprocess(image_pil)
    orig_shape = original_image.shape[:2]
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # Run inference
    # Load ONNX model
    session = ort.InferenceSession("best.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    preds = session.run(None, {input_name: input_tensor})[0]

    # Postprocess
    boxes, scores, class_ids = postprocess(preds, orig_shape, INPUT_SIZE)
    result_img, counts = draw_boxes(original_image.copy(), boxes, scores, class_ids)
    # Postprocessing
    boxes, scores, class_ids = postprocess(preds, image_np.shape[:2])
    result_img, counts = draw_boxes(image_np.copy(), boxes, scores, class_ids)

    # Convert BGR → RGB for Streamlit
    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    st.image(result_img_rgb, caption="📷 Detection Result", use_container_width=True)
    # Show results
    st.image(result_img, caption="🎯 Detection Result", use_container_width=True)

    # Class-wise summary
    st.subheader("📊 Detection Summary")
    st.subheader("📊 Class Counts")
    for cls, count in counts.items():
        if count > 0:
            st.write(f"- **{cls}**: {count}")