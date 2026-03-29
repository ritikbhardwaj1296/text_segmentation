import os
from ultralytics import YOLO

def train_custom_yolo_world(dataset_yaml, experiment_name):
    print(f"\n--- Starting YOLO-World Fine-Tuning for: {experiment_name} ---")
    
    model = YOLO("yolov8x-worldv2.pt")

    results = model.train(
        data=dataset_yaml,      
        epochs=30,              
        imgsz=640,              
        batch=32,               
        device=0,               
        patience=15,            
        name=experiment_name,   
        exist_ok=True          
    )
    
    print(f"\n✅ Training Complete! Best weights saved to runs/detect/{experiment_name}/weights/best.pt")

def resume_custom_yolo_world():
    from ultralytics import YOLOWorld

    model = YOLOWorld("runs/detect/yolo_world_combined_v1/weights/last.pt")

    # 2. Resume training
    results = model.train(resume=True)

    print("✅ Training successfully resumed and completed!")

if __name__ == "__main__":
    
    # CRACK_YAML = "data/cracks/data.yaml" # Update if you renamed the folder
    COMBINED_YAML = "data/combined_dataset/data.yaml"
    
    # train_custom_yolo_world(
    #     dataset_yaml=COMBINED_YAML,
    #     experiment_name="yolo_world_combined_v2"
    # )

    resume_custom_yolo_world()