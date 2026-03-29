from ultralytics import YOLO

class YOLOWorldDetector:
    def __init__(self, model_path="yolov8s-world.pt", classes=[""]):
        self.model = YOLO(model_path).to('cuda')
        # self.model.set_classes(classes)

    def predict(self, image, text):
        self.model.set_classes([text])
        results = self.model(image, verbose=False)

        boxes = []
        for r in results:
            for box in r.boxes.xyxy:
                boxes.append(box.cpu().numpy())

        return boxes