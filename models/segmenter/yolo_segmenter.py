import torch
import numpy as np
import cv2
from ultralytics import YOLO

class YOLOSegmenter:
    def __init__(self, model_path="runs/segment/yolo_seg_combined/weights/best.pt"):
        self.model = YOLO(model_path).to('cuda')

    def predict(self, image, boxes, point_coords=None):
        results = self.model(image, verbose=False)
        masks_out = []
        
        if len(results) == 0 or results[0].masks is None:
            for _ in boxes:
                masks_out.append(np.zeros(image.shape[:2], dtype=bool))
            return masks_out

        yolo_masks = results[0].masks.data.cpu().numpy()
        
        orig_h, orig_w = image.shape[:2]
        resized_masks = []
        for m in yolo_masks:
            resized = cv2.resize(m, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            resized_masks.append(resized > 0.5)

        for box in boxes:
            x_min, y_min, x_max, y_max = map(int, box)
            
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(orig_w, x_max), min(orig_h, y_max)
            
            box_filter = np.zeros((orig_h, orig_w), dtype=bool)
            box_filter[y_min:y_max, x_min:x_max] = True
            
            final_box_mask = np.zeros((orig_h, orig_w), dtype=bool)
            
            for m in resized_masks:
                overlap = np.logical_and(m, box_filter)
                if overlap.sum() > 0:
                    final_box_mask = np.logical_or(final_box_mask, overlap)
                    
            masks_out.append(final_box_mask)

        return masks_out