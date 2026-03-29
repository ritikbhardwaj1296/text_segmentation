import os
import json
import shutil
import yaml
from collections import defaultdict

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
OUTPUT_DIR = "data/combined_dataset"

# The clean, natural-language prompts YOLO-World will actually use for training
TARGET_CLASSES = {
    0: "drywall joint",
    1: "crack"
}

# The mapping dictionary: "Messy COCO Name" -> Target YOLO ID
CATEGORY_MAP = {
    # Drywall mappings -> Class 0
    "Drywall-Join": 0,
    "drywall-join": 0,
    
    # Crack mappings -> Class 1
    "crack": 1,
    "NewCracks - v2 2024-05-18 10:54pm": 1,
    "surface crack": 1, # Adding this just in case it appears in the JSON
    "newcracks": 1
}

def setup_directories():
    """Creates the standard YOLO directory structure."""
    for split in ["train", "valid"]:
        os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)

def process_dataset(dataset_path, prefix, split):
    """Parses COCO JSON, maps classes, and moves files to the combined directory."""
    json_path = os.path.join(dataset_path, split, "_annotations.coco.json") # Adjust if filename differs
    
    # Fallback if the file is just named annotations.json
    if not os.path.exists(json_path):
        json_path = os.path.join(dataset_path, split, "annotations.json")
        
    if not os.path.exists(json_path):
        print(f"⚠️ No annotations found for {dataset_path} ({split})")
        return

    print(f"Processing {prefix} ({split})...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 1. Map this specific JSON's category IDs to our master YOLO IDs
    local_id_to_yolo_id = {}
    for cat in data['categories']:
        cat_name = cat['name']
        if cat_name in CATEGORY_MAP:
            local_id_to_yolo_id[cat['id']] = CATEGORY_MAP[cat_name]
        else:
            print(f"  [Warning] Unknown category found: '{cat_name}'. Ignoring.")

    # Group annotations by image_id
    img_to_anns = defaultdict(list)
    for ann in data['annotations']:
        img_to_anns[ann['image_id']].append(ann)

    # 2. Process images and write labels
    for img in data['images']:
        img_id = img['id']
        old_filename = img['file_name']
        img_w = img['width']
        img_h = img['height']

        # Create a unique filename to prevent overwriting
        new_filename = f"{prefix}_{old_filename}"
        new_txt_name = os.path.splitext(new_filename)[0] + ".txt"

        # Paths
        src_img_path = os.path.join(dataset_path, split, old_filename)
        dst_img_path = os.path.join(OUTPUT_DIR, "images", split, new_filename)
        dst_txt_path = os.path.join(OUTPUT_DIR, "labels", split, new_txt_name)

        # Copy the image
        if os.path.exists(src_img_path):
            shutil.copy(src_img_path, dst_img_path)
        else:
            print(f"  [Error] Missing image file: {src_img_path}")
            continue

        # Write the YOLO txt file
        with open(dst_txt_path, 'w') as f:
            for ann in img_to_anns.get(img_id, []):
                local_cat_id = ann['category_id']
                
                # Skip if we don't care about this category
                if local_cat_id not in local_id_to_yolo_id:
                    continue
                    
                yolo_id = local_id_to_yolo_id[local_cat_id]

                # Convert COCO [x_min, y_min, w, h] to YOLO [x_center, y_center, w, h]
                x_min, y_min, w, h = ann['bbox']
                x_center = (x_min + w / 2) / img_w
                y_center = (y_min + h / 2) / img_h
                norm_w = w / img_w
                norm_h = h / img_h

                f.write(f"{yolo_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

def create_yaml():
    """Generates the data.yaml file required by Ultralytics."""
    yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
    
    # YOLO-World will use these exact strings as text prompts
    names_list = [TARGET_CLASSES[0], TARGET_CLASSES[1]]
    
    yaml_content = {
        "train": os.path.abspath(os.path.join(OUTPUT_DIR, "images/train")),
        "val": os.path.abspath(os.path.join(OUTPUT_DIR, "images/valid")),
        "nc": len(TARGET_CLASSES),
        "names": names_list
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
    print(f"\n✅ Created YOLO YAML configuration at: {yaml_path}")

if __name__ == "__main__":
    print("🚀 Starting Dataset Merge and Conversion...")
    setup_directories()

    # Process Drywall (will become class 0)
    process_dataset("data/drywall_Join", prefix="dw", split="train")
    process_dataset("data/drywall_Join", prefix="dw", split="valid")

    # Process Cracks (will become class 1)
    process_dataset("data/cracks", prefix="crk", split="train")
    process_dataset("data/cracks", prefix="crk", split="valid")

    create_yaml()
    print("🎉 Dataset successfully combined!")