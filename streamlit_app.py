# Cell 1: Install dependencies (if needed)
!pip install onnxruntime opencv-python-headless matplotlib numpy

# Cell 2: Imports
import onnxruntime as ort
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Cell 3: Constants
MODEL_PATH = "best.onnx"  # Adjust if needed
IMAGE_PATH = "test.jpg"   # Replace with path to your test image

INPUT_SIZE = 640

# Cell 4: Load ONNX model
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name
print("ONNX Input Name:", input_name)

# Cell 5: Load and preprocess image
original = cv2.imread(IMAGE_PATH)
original_resized = cv2.resize(original, (INPUT_SIZE, INPUT_SIZE))
image = cv2.cvtColor(original_resized, cv2.COLOR_BGR2RGB).astype(np.float32)
image = np.transpose(image, (2, 0, 1))  # HWC to CHW
image = np.expand_dims(image, axis=0)  # Add batch dim
image /= 255.0  # Normalize if trained with normalized input

# Cell 6: Run inference
outputs = session.run(None, {input_name: image})
output = outputs[0]
print("Output shape:", output.shape)

# Cell 7: Display raw predictions
for i, pred in enumerate(output[0]):
    print(f"{i:02d}: {pred[:10]}")  # Show first 10 values of each prediction

# Cell 8: Postprocessing (experimental)
def postprocess(preds, conf_thresh=0.25):
    boxes, scores, class_ids = [], [], []

    for pred in preds:
        if len(pred) == 6:
            x1, y1, x2, y2, conf, cls = pred
        elif len(pred) > 6:
            # YOLOv8 format: x, y, w, h, obj_conf, class_conf_1, class_conf_2, ...
            x, y, w, h, obj_conf = pred[:5]
            class_confs = pred[5:]
            cls = np.argmax(class_confs)
            conf = obj_conf * class_confs[cls]
            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2
        else:
            continue

        if conf > conf_thresh:
            boxes.append([int(x1), int(y1), int(x2), int(y2)])
            scores.append(float(conf))
            class_ids.append(int(cls))

    return boxes, scores, class_ids

# Cell 9: Apply postprocessing
boxes, scores, class_ids = postprocess(output[0], conf_thresh=0.2)
print(f"\nDetections: {len(boxes)}")
print("Classes:", class_ids)

# Cell 10: Draw results
CLASS_NAMES = ['NO mask', 'NOhairnet', 'hairnet', 'mask']  # update if needed

def draw_boxes(image, boxes, scores, class_ids):
    for box, score, cls_id in zip(boxes, scores, class_ids):
        label = f"{CLASS_NAMES[cls_id]}" if cls_id < len(CLASS_NAMES) else f"Class {cls_id}"
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(image, f"{label} {score:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    return image

output_image = draw_boxes(original_resized.copy(), boxes, scores, class_ids)
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.axis(False)
plt.title("Detections")
plt.show()