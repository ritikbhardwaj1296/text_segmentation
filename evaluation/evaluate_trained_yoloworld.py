import os
import time
import csv
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from pycocotools.coco import COCO

from models.detector.yolo_world_detector import YOLOWorldDetector
from models.segmenter.sam_segmenter import SAMSegmenter
from models.segmenter.morphological_segmenter import MorphologicalSegmenter
from models.segmenter.yolo_segmenter import YOLOSegmenter
from pipelines.base_pipeline import BaselinePipeline, HybridPipeline, PreprocessedPipeline, YOLOCascadePipeline

OUTPUT_MASKS_DIR = "evaluation/rubric_masks_yolo_segmenter_exp"
OUTPUT_CSV = "evaluation/detailed_results_full_yolo_trained_yolo_segmenter_exp.csv"

YOLO_WEIGHTS = "runs/detect/yolo_world_combined_v1/weights/best.pt" 
SAM_WEIGHTS = "weights/sam_vit_h_4b8939.pth"
YOLO_SEG_WEIGHTS = "runs/segment/runs/segment/yolo_seg_combined/weights/best.pt"

SEGMENTER_NAME = "SAM"

DATASETS = {
    "drywall": {
        "image_dir": "data/drywall_join/valid",
        "ann_path": "data/drywall_join/valid/annotations.json", 
        "yolo_class_name": "drywall joint",
        "assignment_prompt": "segment taping area"
    },
    "crack": {
        "image_dir": "data/cracks/valid",
        "ann_path": "data/cracks/valid/annotations.json",
        "yolo_class_name": "crack",
        "assignment_prompt": "segment crack"
    }
}

def setup():
    os.makedirs(OUTPUT_MASKS_DIR, exist_ok=True)
    os.makedirs("evaluation", exist_ok=True)
    
    print(f"Loading Fine-Tuned YOLO from {YOLO_WEIGHTS}...")
    detector = YOLOWorldDetector(model_path=YOLO_WEIGHTS)    
    # print("Loading SAM...")
    # segmenter = SAMSegmenter(checkpoint=SAM_WEIGHTS)
    # print("Loading Morphological Segmenter...")
    # segmenter = MorphologicalSegmenter()
    # SEGMENTER_NAME = "Morph"
    
    segmenter = YOLOSegmenter(model_path=YOLO_SEG_WEIGHTS)
    SEGMENTER_NAME = "yolo_seg"
    
    return YOLOCascadePipeline(detector, segmenter)
    # return BaselinePipeline(detector, segmenter)
    # return HybridPipeline(detector, segmenter)
    # return PreprocessedPipeline(detector, segmenter)

def compute_metrics(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    
    pred_sum = pred_mask.sum()
    gt_sum = gt_mask.sum()

    if union == 0 and pred_sum == 0 and gt_sum == 0:
        return 1.0, 1.0 
    
    iou = intersection / union if union > 0 else 0.0
    dice = (2 * intersection) / (pred_sum + gt_sum) if (pred_sum + gt_sum) > 0 else 0.0
    
    return iou, dice

def save_rubric_mask(merged_mask, image_filename, prompt):
    image_id = Path(image_filename).stem
    safe_prompt = prompt.replace(" ", "_")
    mask_filename = f"{image_id}__{safe_prompt}.png"
    save_path = os.path.join(OUTPUT_MASKS_DIR, mask_filename)
    
    binary_mask = (merged_mask * 255).astype(np.uint8)
    cv2.imwrite(save_path, binary_mask)

def run_evaluation():
    pipeline = setup()
    
    total_inference_time = 0.0
    processed_images = 0
    results = {}

    with open(OUTPUT_CSV, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Image_File", "Prompt", "IoU", "Dice", "Inference_Time_sec"])

        for dataset_name, info in DATASETS.items():
            print(f"\n{'='*50}\nEvaluating Dataset: {dataset_name.upper()}\n{'='*50}")
            
            coco = COCO(info["ann_path"])
            img_ids = coco.getImgIds()
            
            iou_list, dice_list = [], []
            
            for img_id in tqdm(img_ids, desc=f"Evaluating {info['assignment_prompt']}"):
                img_info = coco.loadImgs(img_id)[0]
                img_path = os.path.join(info["image_dir"], img_info['file_name'])
                
                if not os.path.exists(img_path):
                    continue
                    
                image = cv2.imread(img_path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                ann_ids = coco.getAnnIds(imgIds=img_id)
                anns = coco.loadAnns(ann_ids)
                gt_mask = np.zeros((img_info['height'], img_info['width']), dtype=bool)
                
                for ann in anns:
                    if 'segmentation' in ann and ann['segmentation']:
                        gt_mask = np.logical_or(gt_mask, coco.annToMask(ann) > 0)
                    else:
                        x, y, w, h = map(int, ann['bbox'])
                        gt_mask[y:y+h, x:x+w] = True
                start_time = time.time()
                boxes, pred_masks = pipeline.run(img_path, image_rgb, info["yolo_class_name"])
                # boxes, pred_masks = pipeline.run(img_path, image_rgb, info["assignment_prompt"])
                # import pdb;pdb.set_trace()

                inf_time = time.time() - start_time
                
                total_inference_time += inf_time
                processed_images += 1
                
                if len(pred_masks) == 0:
                    merged_pred = np.zeros_like(gt_mask, dtype=bool)
                else:
                    merged_pred = np.zeros_like(pred_masks[0], dtype=bool)
                    for m in pred_masks:
                        merged_pred = np.logical_or(merged_pred, m > 0)
                
                if len(merged_pred.shape) > 2:
                    merged_pred = merged_pred[0]

                if info["yolo_class_name"] == "crack" and SEGMENTER_NAME == "SAM":
                    merged_pred_uint8 = merged_pred.astype(np.uint8)
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    dilated_mask = cv2.dilate(merged_pred_uint8, kernel, iterations=2)
                    merged_pred = dilated_mask > 0
                        
                iou, dice = compute_metrics(merged_pred, gt_mask)
                iou_list.append(iou)
                dice_list.append(dice)
                # import pdb;pdb.set_trace()
                save_rubric_mask(merged_pred, img_info['file_name'], info["assignment_prompt"])
                
                writer.writerow([
                    dataset_name, img_info['file_name'], info["assignment_prompt"], 
                    f"{iou:.4f}", f"{dice:.4f}", f"{inf_time:.4f}"
                ])
                f.flush() 

                torch.cuda.empty_cache() 
                
            results[dataset_name] = {
                "mIoU": np.mean(iou_list) if iou_list else 0.0,
                "mDice": np.mean(dice_list) if dice_list else 0.0
            }

    print("\n" + "="*50)
    print("🏆 FINAL EVALUATION REPORT")
    print("="*50)
    for dataset_name, metrics in results.items():
        print(f"Dataset: {dataset_name.upper()}")
        print(f"  - mIoU:  {metrics['mIoU']:.4f}")
        print(f"  - mDice: {metrics['mDice']:.4f}\n")
        
    avg_time = total_inference_time / processed_images if processed_images > 0 else 0
    print(f"⏱️ Average Inference Time per Image: {avg_time:.3f} seconds")
    print(f"📄 Detailed per-image results saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    run_evaluation()