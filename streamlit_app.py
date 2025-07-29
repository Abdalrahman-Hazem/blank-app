import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time

# Class names (edit as needed)
class_names = ['NO mask', 'NOhairnet', 'hairnet', 'mask']

# Load ONNX model using Ultralytics
# model_path = "E:/Work/KSA/Model1/mask_hairnet_v1/weights/best.onnx"
model_path = "best.onnx"

model = YOLO(model_path, task="detect")

# Draw bounding boxes with color and label
def draw_boxes(image, results, conf_threshold=0.25):
    for result in results:
        boxes = result.boxes
        if boxes is not None and boxes.data is not None:
            box_data = boxes.data.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy().astype(int)
            confidences = boxes.conf.cpu().numpy()

            for i in range(len(box_data)):
                conf = confidences[i]
                if conf < conf_threshold:
                    continue
                x1, y1, x2, y2 = box_data[i][:4].astype(int)
                cls_id = class_ids[i]
                label = f"{class_names[cls_id]}: {conf:.2f}"

                color = (0, 255, 0)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

# --- Streamlit UI ---
st.set_page_config(page_title="Mask & Hairnet Detection", layout="wide")
st.title("😷 Mask and Hairnet Detection (YOLOv8 ONNX)")

# Input options
option = st.radio("Choose Input Type:", ["📷 Webcam", "🌐 IP Camera", "🖼 Upload Image"])

# Confidence threshold
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)

# === Upload Image ===
if option == "🖼 Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        results = model(image)
        image = draw_boxes(image, results, conf_threshold)
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Processed Image", use_container_width=True)

# === Webcam or IP Camera ===
elif option in ["📷 Webcam", "🌐 IP Camera"]:
    # IP input for IP camera
    ip_url = ""
    if option == "🌐 IP Camera":
        ip_url = st.text_input("Enter IP camera RTSP/HTTP link:", "rtsp://...")
    run = st.checkbox("Start Camera Stream")

    if run:
        # Use webcam or IP stream
        source = ip_url if option == "🌐 IP Camera" and ip_url.strip() != "" else 0
        cap = cv2.VideoCapture(source)
        stframe = st.empty()
        prev_time = time.time()

        if not cap.isOpened():
            st.error("❌ Failed to open video stream.")
        else:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    st.warning("❗ Failed to read frame.")
                    break

                # Inference and draw
                results = model(frame)
                frame = draw_boxes(frame, results, conf_threshold)

                # FPS
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time)
                prev_time = curr_time
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

            cap.release()