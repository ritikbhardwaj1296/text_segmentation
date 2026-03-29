from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path).to('cuda')

    def predict(self, image, text=None):
        results = self.model(image)
        boxes = []

        for r in results:
            for box in r.boxes.xyxy:
                boxes.append(box.cpu().numpy())

        return boxes