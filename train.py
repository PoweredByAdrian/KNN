from ultralytics import YOLO

model = YOLO("yolov8m.pt")

# Train the model
results = model.train(
    data="data.yaml",  # Path to the dataset config file
    epochs=80,                 # Number of training epochs
    imgsz=640,                 # Image size (resize to 640x640)
    batch=-1,                    # Batch size (adjust for your GPU/CPU)
    val=True,                  # Enable validation
    device=0,
    seed=42,
)

# it is save automatically
