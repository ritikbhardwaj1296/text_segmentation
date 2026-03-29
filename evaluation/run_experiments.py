import os
import csv
# import sys
# sys.path.append("GroundingDINO")

from utils.dataset import CrackDataset
from evaluation.evaluator import Evaluator

from models.detector.dino_detector import DinoDetector
from models.detector.yolo_detector import YOLODetector
from models.detector.yolo_world_detector import YOLOWorldDetector
from models.segmenter.sam_segmenter import SAMSegmenter
from pipelines.base_pipeline import BaselinePipeline




DATASETS = {
    "crack": {
        "image_dir": "data/cracks/valid",
        "ann_path": "data/cracks/valid/_annotations_cracks_coco.json",
        "prompts": ["crack", "wall crack", "surface crack"]
    },
    "drywall": {
        "image_dir": "data/drywall_Join/valid",
        "ann_path": "data/drywall_Join/valid/_annotations.coco.json",
        "prompts": ["drywall joint", "wall joint"]
    }
}

DETECTORS = ["yolo_world"]#, "dino"]

OUTPUT_CSV = "evaluation/results.csv"


def check_device(detector_name, detector, segmenter):
    print("\n--- Verifying GPU Assignment ---")
    try:
        if "yolo" in detector_name:
            # Ultralytics YOLO stores the device as a direct property
            det_device = detector.model.device
        else:
            # DINO requires checking the actual parameter tensors
            det_device = next(detector.model.parameters()).device
            
        print(f"{detector_name.upper()} weights are currently on: {det_device}")
    except Exception as e:
        print(f"Could not verify device for {detector_name}. Reason: {e}")

    # 2. Check SAM
    try:
        sam_device = segmenter.predictor.model.device
        print(f"SAM weights are currently on: {sam_device}")
    except Exception as e:
        print(f"Could not verify device for SAM. Reason: {e}")

    print("--------------------------------\n")

def get_detector(detector_type, classes):

    if detector_type == "dino":
        return DinoDetector(
            config_path="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            weight_path="weights/groundingdino_swint_ogc.pth"
        )

    elif detector_type == "yolo":
        return YOLODetector("yolov8n.pt")

    elif detector_type == "yolo_world":
        return YOLOWorldDetector("yolov8x-worldv2.pt", classes)

    else:
        raise ValueError("Unknown detector")



def run():

    os.makedirs("evaluation", exist_ok=True)

    # prepare CSV
    with open(OUTPUT_CSV, mode="w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "dataset",
            "detector",
            "prompt",
            "iou",
            "dice",
            "precision",
            "recall"
        ])

        # segmenter = SAMSegmenter()
        segmenter = SAMSegmenter(checkpoint="weights/sam_vit_h_4b8939.pth")

        # loop over datasets
        for dataset_name, dataset_info in DATASETS.items():

            print(f"\n===== DATASET: {dataset_name} =====")

            dataset = CrackDataset(
                image_dir=dataset_info["image_dir"],
                annotation_path=dataset_info["ann_path"]
            )

            # loop over detectors
            for detector_name in DETECTORS:

                print(f"\n--- Detector: {detector_name} ---")

                detector = get_detector(detector_name, dataset_info["prompts"])
                #Verify device GPU vs CPU
                check_device(detector_name, detector, segmenter)

                pipeline = BaselinePipeline(detector, segmenter)

                evaluator = Evaluator(pipeline)

                # loop over prompts
                for prompt in dataset_info["prompts"]:

                    print(f"Running prompt: {prompt}")

                    results = evaluator.evaluate_dataset(dataset, prompt)

                    print(results)

                    writer.writerow([
                        dataset_name,
                        detector_name,
                        prompt,
                        results["iou"],
                        results["dice"],
                        results["precision"],
                        results["recall"]
                    ])

    print(f"\n✅ Results saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    run()