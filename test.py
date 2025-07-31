import cv2
import numpy as np
import onnxruntime as ort

# ✅ Class names as per data.yaml
CLASS_NAMES = ['NO mask', 'NOhairnet', 'hairnet', 'mask']
CONFIDENCE_THRESHOLD = 0.5

def preprocess(image, size=640):
    image_resized = cv2.resize(image, (size, size))
    image_input = image_resized.transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32) / 255.0
    return image_input

def postprocess(predictions, orig_shape, input_shape):
    boxes, scores, class_ids = [], [], []
    num_classes = len(CLASS_NAMES)
    for pred in predictions:
        x1, y1, x2, y2 = pred[:4]
        cls_scores = pred[4:]
        class_id = int(np.argmax(cls_scores))
        confidence = cls_scores[class_id]
        if confidence > CONFIDENCE_THRESHOLD and class_id < num_classes:
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

# ✅ Load image
image_path = "test3.jpg"
original = cv2.imread(image_path)

# ✅ Load ONNX model
session = ort.InferenceSession("best.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
input_shape = (640, 640)

# ✅ Preprocess
image_input = preprocess(original, size=640)
outputs = session.run(None, {input_name: image_input})[0]

# ✅ Postprocess
outputs = outputs.squeeze().T  # Shape: (8400, 8)
boxes, scores, class_ids = postprocess(outputs, original.shape[:2], input_shape)

# ✅ Draw and save
result_img, class_counts = draw_boxes(original.copy(), boxes, scores, class_ids)
cv2.imwrite("result.jpg", result_img)

print(f"✅ ONNX Input Name: {input_name}")
print(f"✅ Transposed output shape: {outputs.shape}")
print(f"✅ Detections: {len(boxes)}")
for cls, count in class_counts.items():
    if count > 0:
        print(f"{cls}: {count}")
print("✅ Saved result image as result.jpg")