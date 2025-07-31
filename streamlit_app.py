import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import Counter

# Load the ONNX-exported YOLOv8 model
model_path = "model.onnx"
model = YOLO(model_path, task="detect")

# Define the classes (based on your model)
class_names = ['NO mask', 'NOhairnet', 'hairnet', 'mask']
colors = [(255, 0, 0), (0, 255, 0), (0, 255, 255), (255, 255, 0)]

# Draw boxes and labels
def draw_boxes(frame, results):
    if results.boxes:
        for box in results.boxes:
            cls_id = int(box.cls.cpu().numpy()[0])
            conf = float(box.conf.cpu().numpy()[0])
            x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])

            color = colors[cls_id % len(colors)]
            label = f"{class_names[cls_id]} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

# Count class occurrences
def count_classes(results):
    counts = Counter()
    if results.boxes:
        for box in results.boxes:
            cls_id = int(box.cls.cpu().numpy()[0])
            counts[class_names[cls_id]] += 1
    return counts

# Streamlit UI
st.title("Mask and Hairnet Detection")
option = st.radio("Choose input type", ("Webcam", "IP Camera", "Upload Image"))

confidence = st.slider("Confidence Threshold", 0.25, 1.0, 0.5, 0.05)

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

        results = model(frame, conf=confidence)[0]
        frame = draw_boxes(frame, results)

        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", caption="Processed Image")

        counts = count_classes(results)
        st.sidebar.subheader("Detections")
        for cls, val in counts.items():
            st.sidebar.write(f"{cls}: {val}")

elif option in ("Webcam", "IP Camera"):
    ip_url = ""
    if option == "IP Camera":
        ip_url = st.text_input("Enter RTSP/HTTP URL", "")

    run = st.checkbox("Start")
    stop = st.button("Stop")

    cap = None
    if run:
        source = 0 if option == "Webcam" else ip_url
        cap = cv2.VideoCapture(source)

        stframe = st.empty()

        while cap.isOpened() and not stop:
            success, frame = cap.read()
            if not success:
                st.warning("Failed to read from source.")
                break

            frame = cv2.resize(frame, (640, 480))  # Optional for performance
            results = model(frame, conf=confidence)[0]
            frame = draw_boxes(frame, results)

            counts = count_classes(results)
            st.sidebar.subheader("Detections")
            for cls, val in counts.items():
                st.sidebar.write(f"{cls}: {val}")

            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        cap.release()
        st.success("Stream stopped.")