from ultralytics import YOLO

model = YOLO("E:/Work/KSA/Model1/mask_hairnet_v1/weights/best.pt")
results = model("test3.jpg", conf=0.2)

# Iterate over the list of result(s)
for result in results:
    result.show()      # Display image with detections
    result.save(filename="test3_output.jpg")  # Save if needed
    result.print()      # Print class info