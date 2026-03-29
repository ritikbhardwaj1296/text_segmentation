import argparse
from utils.dataset import CrackDataset
from evaluation.evaluator import Evaluator

from models.detector.dino_detector import DinoDetector
from models.detector.yolo_detector import YOLODetector
from models.detector.yolo_world_detector import YOLOWorldDetector
from models.segmenter.sam_segmenter import SAMSegmenter
from pipelines.baseline_pipeline import BaselinePipeline


def get_detector(detector_type):
    if detector_type == "dino":
        return DinoDetector(
            config_path="GroundingDINO/config.py",
            weight_path="weights/groundingdino.pth"
        )
    elif detector_type == "yolo_world":
        return YOLOWorldDetector("yolov8s-world.pt")
    elif detector_type == "yolo":
        return YOLODetector("yolov8n.pt")
    else:
        raise ValueError("Unknown detector")


def main(args):

    # dataset
    dataset = CrackDataset(
        image_dir=args.image_dir,
        annotation_path=args.ann_path
    )

    # models
    detector = get_detector(args.detector)
    segmenter = SAMSegmenter()

    pipeline = BaselinePipeline(detector, segmenter)

    evaluator = Evaluator(pipeline)

    results = evaluator.evaluate_dataset(dataset, args.text_prompt)

    print("\n===== RESULTS =====")
    print(f"Detector: {args.detector}")
    print(f"Prompt: {args.text_prompt}")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--ann_path", type=str, required=True)
    parser.add_argument("--detector", type=str, choices=["dino", "yolo"], required=True)
    parser.add_argument("--text_prompt", type=str, default="crack")

    args = parser.parse_args()

    main(args)