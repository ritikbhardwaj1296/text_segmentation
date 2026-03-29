from ultralytics import YOLO

def train_yolo_segmentation(experiment_name):
    print(f"\n--- Starting YOLO-SEF Fine-Tuning for: {experiment_name} ---")
    model = YOLO("yolov8x-seg.pt")
    
    results = model.train(
        data="data/yolo_seg_dataset/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        patience=15,
        # pin_memory=False,
        project="runs/segment",
        name=experiment_name,
        exist_ok=True
    )

if __name__ == "__main__":
    train_yolo_segmentation("yolo_seg_combined")