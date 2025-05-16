from ultralytics import YOLO

model = YOLO("runs/detect/train5/weights/best.pt")

metrics = model.val(data="data.yaml", split="test",imgsz=640,plots=True)
