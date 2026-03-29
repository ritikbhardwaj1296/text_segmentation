# # from models.detector.dino_detector import DinoDetector
# from models.detector.yolo_world_detector import YOLOWorldDetector
# from models.segmenter.sam_segmenter import SAMSegmenter
# from pipelines.base_pipeline import BaselinePipeline, YOLOCascadePipeline
# from models.segmenter.yolo_segmenter import YOLOSegmenter
# from utils.visualization import visualize_predictions
# import cv2

# img_path = "data/cracks/valid/Image_-642-_jpg.rf.fMM6VyGi3M9qp5XbOMQH.jpg"
# img_name = img_path.split("/")[-1]
# image = cv2.imread(img_path)


# detector = YOLOWorldDetector("runs/detect/yolo_world_combined_v1/weights/best.pt", ["drywall joint", "crack"])
# model_name = "yolo_world"

# YOLO_SEG_WEIGHTS = "runs/segment/runs/segment/yolo_seg_combined/weights/best.pt"
# segmenter = YOLOSegmenter(model_path=YOLO_SEG_WEIGHTS)
# SEGMENTER_NAME = "yolo_seg"
    
# pipeline = YOLOCascadePipeline(detector, segmenter)

# # pipeline = BaselinePipeline(detector, segmenter)

# prompt = "cracks"
# boxes, masks = pipeline.run(img_path, image, prompt)

# if model_name == 'dino':
#     masks = masks[0][0]

# save_path = f"pipelines/vis_save/{model_name}_{img_name}"
# # import pdb;pdb.set_trace()
# # visualize_predictions(image, boxes, masks, prompt, save_path=save_path)

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from models.detector.yolo_world_detector import YOLOWorldDetector
# from models.segmenter.sam_segmenter import SAMSegmenter
from models.segmenter.yolo_segmenter import YOLOSegmenter
from pipelines.base_pipeline import BaselinePipeline, YOLOCascadePipeline

YOLO_WEIGHTS = "runs/detect/yolo_world_combined_v1/weights/best.pt"
YOLO_SEG_WEIGHTS = "runs/segment/runs/segment/yolo_seg_combined/weights/best.pt"

def setup_pipeline():
    detector = YOLOWorldDetector(model_path=YOLO_WEIGHTS)
    # segmenter = SAMSegmenter(checkpoint=SAM_WEIGHTS)
    segmenter = YOLOSegmenter(model_path=YOLO_SEG_WEIGHTS)
    return YOLOCascadePipeline(detector, segmenter)

def draw_overlay(image, boxes, mask, box_color=(255, 0, 0), mask_color=(255, 0, 0)):
    overlay = image.copy()
    
    if mask is not None and np.any(mask):
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0] = mask_color
        cv2.addWeighted(colored_mask, 0.5, overlay, 0.5, 0, overlay)
        
    for box in boxes:
        x_min, y_min, x_max, y_max = map(int, box)
        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), box_color, 2)
        
    return overlay

def infer_single_image(image_path, prompt, output_path="single_inference.png"):
    pipeline = setup_pipeline()
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
        
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    prompt_mapping = {"segment crack": "crack", "segment taping area": "drwall joints"}

    mapped_prompt = prompt_mapping[PROMPT.lower()]

    pred_boxes, pred_masks_list = pipeline.run(image_path, image_rgb, mapped_prompt)
    
    if len(pred_masks_list) > 0:
        if isinstance(pred_masks_list, list):
            pred_mask = np.zeros_like(pred_masks_list[0], dtype=bool)
            for m in pred_masks_list:
                pred_mask = np.logical_or(pred_mask, m > 0)
        else:
            pred_mask = np.zeros(image_rgb.shape[:2], dtype=bool)
            for m in pred_masks_list:
                pred_mask = np.logical_or(pred_mask, m > 0)
    else:
        pred_mask = np.zeros(image_rgb.shape[:2], dtype=bool)
    
    # import pdb;pdb.set_trace()
    pred_overlay = draw_overlay(image_rgb, pred_boxes, pred_mask, box_color=(255, 0, 0), mask_color=(255, 0, 0))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"Single Inference | Prompt: '{prompt}'", fontsize=16)
    
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(pred_overlay)
    axes[1].set_title("Predicted BBox & Mask")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved inference result to {output_path}")

if __name__ == "__main__":
    model_name = "YOLO"
    TEST_IMAGE = "data/cracks/valid/Image_-642-_jpg.rf.fMM6VyGi3M9qp5XbOMQH.jpg"
    # TEST_IMAGE = "data/cracks/valid/image12_jpg.rf.IgZenjIculNpCRUMkbNe.jpg"
    img_name = TEST_IMAGE.split("/")[-1]
    PROMPT = "Segment Crack"

    # OUT_IMAGE = "pipelines/vis_save/single_inference_result.png"
    OUT_IMAGE = f"pipelines/vis_save/{model_name}_{img_name}"
    
    os.makedirs("evaluation", exist_ok=True)
    infer_single_image(TEST_IMAGE, PROMPT, OUT_IMAGE)
