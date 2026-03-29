import torch
from segment_anything import sam_model_registry, SamPredictor
import numpy as np

class SAMSegmenter:
    def __init__(self, checkpoint="sam_vit_h.pth"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry["vit_h"](checkpoint=checkpoint)
        sam.to(device=self.device)
        self.predictor = SamPredictor(sam)

    def predict(self, image, boxes):
        self.predictor.set_image(image)
        masks = []

        for box in boxes:
            mask, _, _ = self.predictor.predict(
                box=np.array(box),
                multimask_output=False
            )
            masks.append(mask)

        return masks
    

    def predict_with_points(self, image, boxes, point_coords=None):
        self.predictor.set_image(image)
        masks = []
        
        input_boxes = torch.tensor(boxes, device=self.predictor.device)
        transformed_boxes = self.predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
        
        for i in range(len(transformed_boxes)):
            box = transformed_boxes[i].unsqueeze(0)
            
            pt = None
            pt_label = None
            if point_coords is not None and point_coords[i] is not None:
                pt = np.array([point_coords[i]]) # Shape (1, 2)
                pt_label = np.array([1])         # 1 means "foreground"
            
            mask, _, _ = self.predictor.predict_torch(
                point_coords=pt if pt is None else torch.tensor(pt, device=self.device).unsqueeze(0),
                point_labels=pt_label if pt_label is None else torch.tensor(pt_label, device=self.device).unsqueeze(0),
                boxes=box,
                multimask_output=False
            )
            masks.append(mask.squeeze().cpu().numpy())
            
        return masks