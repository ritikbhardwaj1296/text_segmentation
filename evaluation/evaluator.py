import numpy as np
from tqdm import tqdm
from utils.metrics import compute_iou, compute_dice, compute_precision_recall

class Evaluator:

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def evaluate_sample(self, image_path, image, gt_masks, text_prompt):
        # import pdb;pdb.set_trace()
        boxes, pred_masks = self.pipeline.run(image_path, image, text_prompt)

        if len(pred_masks) == 0 or len(gt_masks) == 0:
            return None

        # merge predicted masks
        pred_mask = np.zeros_like(gt_masks[0])
        for m in pred_masks:
            pred_mask = np.logical_or(pred_mask, m)

        # merge GT masks
        gt_mask = np.zeros_like(gt_masks[0])
        for m in gt_masks:
            gt_mask = np.logical_or(gt_mask, m)

        iou = compute_iou(pred_mask, gt_mask)
        dice = compute_dice(pred_mask, gt_mask)
        precision, recall = compute_precision_recall(pred_mask, gt_mask)

        return {
            "iou": iou,
            "dice": dice,
            "precision": precision,
            "recall": recall
        }

    def evaluate_dataset(self, dataset, text_prompt):

        results = []

        # for i in range(len(dataset)):
        for i in tqdm(range(len(dataset)), desc=f"Evaluating '{text_prompt}'"):
            # print(f"Evaluation sample {i} out of total samples {len(dataset)}", end="\r")
            image_path, image, _, gt_masks = dataset[i]

            metrics = self.evaluate_sample(image_path, image, gt_masks, text_prompt)

            if metrics is not None:
                results.append(metrics)

        return self.aggregate(results)

    def aggregate(self, results):
        keys = results[0].keys()

        agg = {}
        for k in keys:
            agg[k] = np.mean([r[k] for r in results])

        return agg