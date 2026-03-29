import torch
from groundingdino.util.inference import load_model, predict, load_image

class DinoDetector:
    def __init__(self, config_path, weight_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = load_model(config_path, weight_path).to(self.device)

    def predict(self, image, text):
        image_source, image_tensor = load_image(image)
        boxes, logits, phrases = predict(
            model=self.model,
            image=image_tensor,
            caption=text,
            box_threshold=0.3,
            text_threshold=0.25
        )
        return boxes