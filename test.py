import onnxruntime as ort
import numpy as np
import cv2
import matplotlib.pyplot as plt

# --- Constants ---
MODEL_PATH = "best.onnx"
IMAGE_PATH = "test2.jpg"  # Replace with your image path
CLASS_NAMES = ['NO mask', 'NOhairnet', 'hairnet', 'mask']
INPUT_SIZE = 640
CONFIDENCE_THRESHOLD = 0.1  # Lowered to help debug

# --- Load Model ---
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name
print("✅ ONNX Input Name:", input_name)

# --- Load & Preprocess Image ---
original = cv2.imread(IMAGE_PATH)
if original is None:
    raise FileNotFoundError(f"❌ Failed to load image: {IMAGE_PATH}")

original_h, original_w = original.shape[:2]
resized = cv2.resize(original, (INPUT_SIZE, INPUT_SIZE))
image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
image = np.transpose(image, (2, 0, 1))  # HWC to CHW
image = np.expand_dims(image, axis=0)

# --- Inference ---
outputs = session.run(None, {input_name: image})
preds = np.squeeze(outputs[0]).T  # (8400, 8)
print("✅ Output shape:", preds.shape)

# --- Postprocess ---
def postprocess(preds, input_shape=(640, 640), orig_shape=(0, 0), conf_thresh=0.1):
    boxes, scores, class_ids = [], [], []
    input_h, input_w = input_shape
    orig_h, orig_w = orig_shape

    for i, pred in enumerate(preds):
        if len(pred) < 6:
            continue

        x, y, w, h, obj_conf = pred[:5]
        class_probs = pred[5:]
        cls = int(np.argmax(class_probs))
        conf = obj_conf * class_probs[cls]

        if i < 20:  # Print first 20 predictions for debug
            print(f"[{i:04}] class={cls}, obj_conf={obj_conf:.4f}, class_conf={class_probs[cls]:.4f}, final_conf={conf:.4f}")

        if conf < conf_thresh:
            continue

        # Convert to x1, y1, x2, y2 format
        x1 = int((x - w / 2) / input_w * orig_w)
        y1 = int((y - h / 2) / input_h * orig_h)
        x2 = int((x + w / 2) / input_w * orig_w)
        y2 = int((y + h / 2) / input_h * orig_h)

        boxes.append([x1, y1, x2, y2])
        scores.append(conf)
        class_ids.append(cls)

    return boxes, scores, class_ids

# --- Run Postprocess ---
boxes, scores, class_ids = postprocess(
    preds,
    input_shape=(INPUT_SIZE, INPUT_SIZE),
    orig_shape=(original_h, original_w),
    conf_thresh=CONFIDENCE_THRESHOLD
)

print(f"\n✅ Detections: {len(boxes)}")
print("🎯 Classes:", [CLASS_NAMES[i] if i < len(CLASS_NAMES) else i for i in class_ids])

# --- Draw Boxes ---
def draw_boxes(image, boxes, scores, class_ids):
    for box, score, cls_id in zip(boxes, scores, class_ids):
        label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"Class {cls_id}"
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(image, f"{label} {score:.2f}", (x1, max(y1 - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    return image

output_image = draw_boxes(original.copy(), boxes, scores, class_ids)
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.title("Detections")
plt.axis(False)
plt.tight_layout()
plt.show()