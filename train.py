from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# Train the model
results = model.train(
    data="dataset/data.yaml",  # Path to the dataset config file
    epochs=50,                 # Number of training epochs
    imgsz=640,                 # Image size (resize to 640x640)
    batch=8,                    # Batch size (adjust for your GPU/CPU)
    val=True,                  # Enable validation
    split=0.8                   # 80% training, 20% validation (YOLO does the split)
)

# Save the trained model
model.save("yolo_trained.pt")
