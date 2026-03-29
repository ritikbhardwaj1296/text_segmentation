import os
import csv
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from pycocotools.coco import COCO

from models.detector.yolo_world_detector import YOLOWorldDetector
# from models.segmenter.morphological_segmenter import MorphologicalSegmenter
from models.segmenter.sam_segmenter import SAMSegmenter
from pipelines.base_pipeline import BaselinePipeline

CSV_PATH = "evaluation/detailed_results_full_yolo_trained_yolo_segmenter.csv"
MASKS_DIR = "evaluation/rubric_masks_yolo_segmenter"
ALL_VIS_DIR = "evaluation/all_visualizations"
BEST_CASES_DIR = "evaluation"

YOLO_WEIGHTS = "runs/detect/yolo_world_combined_v1/weights/best.pt" 
SAM_WEIGHTS = "weights/sam_vit_h_4b8939.pth"

DATASETS = {
    "drywall": {
        "image_dir": "data/drywall_join/valid",
        "ann_path": "data/drywall_join/valid/annotations.json",
        "yolo_class": "drywall joint"
    },
    "crack": {
        "image_dir": "data/cracks/valid",
        "ann_path": "data/cracks/valid/annotations.json",
        "yolo_class": "crack"
    }
}

def setup_pipeline():
    print(f"Loading YOLO from {YOLO_WEIGHTS}...")
    detector = YOLOWorldDetector(model_path=YOLO_WEIGHTS) #, is_finetuned=True)
    detector.model.set_classes(["drywall joint", "crack"]) 
    # print("Loading Morphological Segmenter...")
    # segmenter = MorphologicalSegmenter()
    print("Loading SAM...")
    segmenter = SAMSegmenter(checkpoint=SAM_WEIGHTS)
    return BaselinePipeline(detector, segmenter)

def draw_overlay(image, boxes, mask, box_color=(255, 0, 0), mask_color=(255, 0, 0)):
    overlay = image.copy()
    
    if mask is not None and np.any(mask):
        colored_mask = np.zeros_like(image)
        # import pdb;pdb.set_trace()
        colored_mask[mask > 0] = mask_color
        cv2.addWeighted(colored_mask, 0.5, overlay, 0.5, 0, overlay)
        
    for box in boxes:
        x_min, y_min, x_max, y_max = map(int, box)
        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), box_color, 2)
        
    return overlay


def plot_best_cases(limit=5):
    print(f"\n🌟 Generating Top {limit} Success Cases Grid...")
    results = []
    
    if not os.path.exists(CSV_PATH):
        print("⚠️ CSV not found. Run validation first!")
        return
        
    with open(CSV_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['Dataset'] == 'crack':
                results.append(row)
            # results.append(row)
            
    results.sort(key=lambda x: float(x['IoU']), reverse=True)
    best_images = results[limit:limit+5]
    
    fig, axes = plt.subplots(len(best_images), 3, figsize=(15, 4 * len(best_images)))
    fig.suptitle("Top 5 Success Cases (Orig | Ground Truth | Prediction)", fontsize=22)

    for i, row in enumerate(best_images):
        dataset = row['Dataset'].lower()
        img_filename = row['Image_File']
        prompt = row['Prompt']
        
        img_path = os.path.join(DATASETS[dataset]["image_dir"], img_filename)
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        
        coco = COCO(DATASETS[dataset]["ann_path"])
        img_id = [img['id'] for img in coco.dataset['images'] if img['file_name'] == img_filename][0]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        img_info = coco.loadImgs(img_id)[0]
        
        gt_mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        for ann in anns:
            if 'segmentation' in ann and ann['segmentation']:
                gt_mask = np.logical_or(gt_mask, coco.annToMask(ann)).astype(np.uint8)
            else:
                x, y, w, h = map(int, ann['bbox'])
                gt_mask[y:y+h, x:x+w] = 1
                
        safe_prompt = prompt.replace(" ", "_")
        pred_filename = f"{os.path.splitext(img_filename)[0]}__{safe_prompt}.png"
        pred_path = os.path.join(MASKS_DIR, pred_filename)
        pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

        ax_orig = axes[i, 0] if len(best_images) > 1 else axes[0]
        ax_gt = axes[i, 1] if len(best_images) > 1 else axes[1]
        ax_pred = axes[i, 2] if len(best_images) > 1 else axes[2]

        ax_orig.imshow(img)
        ax_orig.set_title(f"Original: {img_filename}\nIoU: {row['IoU']}")
        ax_orig.axis('off')

        ax_gt.imshow(gt_mask, cmap='gray')
        ax_gt.set_title("Ground Truth")
        ax_gt.axis('off')

        ax_pred.imshow(pred_mask, cmap='gray')
        ax_pred.set_title("Pipeline Prediction")
        ax_pred.axis('off')

    plt.tight_layout()
    save_path = os.path.join(BEST_CASES_DIR, "success_cases_grid_.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"✅ Saved success cases grid to {save_path}")


def generate_all_visualizations():
    os.makedirs(ALL_VIS_DIR, exist_ok=True)
    pipeline = setup_pipeline()
    
    for dataset_name, info in DATASETS.items():
        print(f"\n🔍 Generating 4-Panel visuals for: {dataset_name.upper()}")
        
        coco = COCO(info["ann_path"])
        img_ids = coco.getImgIds()
        
        dataset_out_dir = os.path.join(ALL_VIS_DIR, dataset_name)
        os.makedirs(dataset_out_dir, exist_ok=True)
        
        for img_id in tqdm(img_ids, desc="Processing Images"):
            img_info = coco.loadImgs(img_id)[0]
            img_path = os.path.join(info["image_dir"], img_info['file_name'])
            
            if not os.path.exists(img_path):
                continue
                
            image = cv2.imread(img_path)
            if image is None: continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            
            gt_mask = np.zeros((img_info['height'], img_info['width']), dtype=bool)
            gt_boxes = []
            
            for ann in anns:
                x, y, w, h = ann['bbox']
                gt_boxes.append([x, y, x+w, y+h])
                
                if 'segmentation' in ann and ann['segmentation']:
                    gt_mask = np.logical_or(gt_mask, coco.annToMask(ann) > 0)
                else:
                    x, y, w, h = map(int, ann['bbox'])
                    gt_mask[y:y+h, x:x+w] = True
                    
            pred_boxes, pred_masks_list = pipeline.run(img_path, image_rgb, info["yolo_class"])
            
            pred_mask = np.zeros_like(gt_mask, dtype=bool)
            for m in pred_masks_list:
                pred_mask = np.logical_or(pred_mask, m > 0)
                

            gt_overlay = draw_overlay(image_rgb, gt_boxes, gt_mask, box_color=(0, 255, 0), mask_color=(0, 255, 0))
            pred_overlay = draw_overlay(image_rgb, pred_boxes, pred_mask[0], box_color=(255, 0, 0), mask_color=(255, 0, 0))
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            fig.suptitle(f"File: {img_info['file_name']} | Class: {info['yolo_class']}", fontsize=14)
            
            axes[0, 0].imshow(image_rgb)
            axes[0, 0].set_title("1. Original Image")
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(gt_overlay)
            axes[0, 1].set_title("2. GT BBox & Mask")
            axes[0, 1].axis('off')
            
            axes[1, 0].imshow(pred_overlay)
            axes[1, 0].set_title("3. Predicted BBox & Mask")
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(image_gray, cmap='gray')
            axes[1, 1].set_title("4. Grayscale Base")
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            
            save_path = os.path.join(dataset_out_dir, f"vis_{img_info['file_name']}")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

if __name__ == "__main__":
    # plot_best_cases(limit=5)
    
    generate_all_visualizations()
    
    print("\n✅ All visualizations complete!")