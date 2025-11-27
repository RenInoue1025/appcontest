from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="combined_dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="desk_paper_book"
)
