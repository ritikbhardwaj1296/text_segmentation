import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

CSV_PATH = "evaluation/detailed_results_full_yolo_trained_yolo_segmenter.csv"
MASKS_DIR = "evaluation/rubric_masks_yolo_segmenter"
DATASETS = {
    "drywall": {"image_dir": "data/drywall_join/valid", "ann_path": "data/drywall_join/valid/annotations.json"},
    "crack": {"image_dir": "data/cracks/valid", "ann_path": "data/cracks/valid/annotations.json"}
}

def load_worst_performers(limit=5):
    results = []
    with open(CSV_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
        # if row['Dataset'] == 'crack': # For best cases
        #         results.append(row)
            
    # Sort by IoU ascending
    results.sort(key=lambda x: float(x['IoU']))
    # results.sort(key=lambda x: float(x['IoU']), reverse=True) # For best cases
    return results[:limit]

def plot_failures():
    worst_images = load_worst_performers(5)
    
    fig, axes = plt.subplots(len(worst_images), 3, figsize=(15, 4 * len(worst_images)))
    fig.suptitle("Top 5 Failure Cases (Orig | Ground Truth | Prediction)", fontsize=22)

    for i, row in enumerate(worst_images):
        dataset = row['Dataset'].lower()
        img_filename = row['Image_File']
        prompt = row['Prompt']
        
        # 1. Load Original Image
        img_path = os.path.join(DATASETS[dataset]["image_dir"], img_filename)
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        
        # 2. Reconstruct Ground Truth
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
                
        # 3. Load Prediction Mask
        safe_prompt = prompt.replace(" ", "_")
        pred_filename = f"{os.path.splitext(img_filename)[0]}__{safe_prompt}.png"
        pred_path = os.path.join(MASKS_DIR, pred_filename)
        pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

        # Plotting
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Original: {img_filename}\nIoU: {row['IoU']}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(gt_mask, cmap='gray')
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis('off')

        axes[i, 2].imshow(pred_mask, cmap='gray')
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig("evaluation/failure_cases_grid.png")
    print("✅ Saved failure cases grid to evaluation/failure_cases_grid.png")

if __name__ == "__main__":
    plot_failures()