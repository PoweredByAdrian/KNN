from ultralytics import YOLO

model = YOLO("../train3/weights/best.pt")

metrics = model.val(data="data.yaml", split="test",imgsz=640,plots=True)
