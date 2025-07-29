import cv2
import torch
from ultralytics import YOLO

# Load trained ONNX model
model = YOLO("E:/Work/KSA/Model1/mask_hairnet_v1/weights/best.onnx")

device = "cuda" if torch.cuda.is_available() else "cpu"


def detect_from_camera(source=0):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ Failed to open video source: {source}")
        return

    print("✅ Press 'q' to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection
        results = model.predict(source=frame, imgsz=640, conf=0.4, device=device)
        result_frame = results[0].plot()

        # Show results
        cv2.imshow("YOLOv8 Detection", result_frame)

        # Quit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # webcam:
    detect_from_camera(0)

    # RTSP stream:
    # detect_from_camera("rtsp://username:password@ip_address:port")