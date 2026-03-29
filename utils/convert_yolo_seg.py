import os
import json
import shutil
import yaml
from collections import defaultdict

OUTPUT_DIR = "data/yolo_seg_dataset"
TARGET_CLASSES = {0: "drywall joint", 1: "crack"}
CATEGORY_MAP = {
    "Drywall-Join": 0, "drywall-join": 0,
    "crack": 1, "NewCracks - v2 2024-05-18 10:54pm": 1, "surface crack": 1, "newcracks": 1
}

def setup_dirs():
    for split in ["train", "valid"]:
        os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, "labels", split), exist_ok=True)

def process_seg_dataset(dataset_path, prefix, split):
    json_path = os.path.join(dataset_path, split, "_annotations.coco.json")
    if not os.path.exists(json_path):
        json_path = os.path.join(dataset_path, split, "annotations.json")
    if not os.path.exists(json_path): return

    with open(json_path, 'r') as f:
        data = json.load(f)

    local_to_yolo = {cat['id']: CATEGORY_MAP[cat['name']] for cat in data['categories'] if cat['name'] in CATEGORY_MAP}
    img_to_anns = defaultdict(list)
    for ann in data['annotations']:
        img_to_anns[ann['image_id']].append(ann)

    for img in data['images']:
        img_id = img['id']
        old_filename = img['file_name']
        img_w, img_h = img['width'], img['height']

        new_filename = f"{prefix}_{old_filename}"
        new_txt_name = os.path.splitext(new_filename)[0] + ".txt"

        src_img = os.path.join(dataset_path, split, old_filename)
        dst_img = os.path.join(OUTPUT_DIR, "images", split, new_filename)
        dst_txt = os.path.join(OUTPUT_DIR, "labels", split, new_txt_name)

        if not os.path.exists(src_img):
            continue

        shutil.copy(src_img, dst_img)

        with open(dst_txt, 'w') as f:
            for ann in img_to_anns.get(img_id, []):
                local_cat_id = ann['category_id']
                if local_cat_id not in local_to_yolo: continue
                yolo_id = local_to_yolo[local_cat_id]

                if 'segmentation' in ann and ann['segmentation']:
                    for poly in ann['segmentation']:
                        if len(poly) >= 6:
                            norm_poly = [val / img_w if i % 2 == 0 else val / img_h for i, val in enumerate(poly)]
                            f.write(f"{yolo_id} " + " ".join([f"{x:.6f}" for x in norm_poly]) + "\n")
                else:
                    x, y, w, h = ann['bbox']
                    pts = [x, y, x+w, y, x+w, y+h, x, y+h]
                    norm_poly = [val / img_w if i % 2 == 0 else val / img_h for i, val in enumerate(pts)]
                    f.write(f"{yolo_id} " + " ".join([f"{v:.6f}" for v in norm_poly]) + "\n")

def create_yaml():
    yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
    yaml_content = {
        "train": os.path.abspath(os.path.join(OUTPUT_DIR, "images/train")),
        "val": os.path.abspath(os.path.join(OUTPUT_DIR, "images/valid")),
        "nc": len(TARGET_CLASSES),
        "names": [TARGET_CLASSES[0], TARGET_CLASSES[1]]
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)

if __name__ == "__main__":
    setup_dirs()
    process_seg_dataset("data/drywall_join", "dw", "train")
    process_seg_dataset("data/drywall_join", "dw", "valid")
    process_seg_dataset("data/cracks", "crk", "train")
    process_seg_dataset("data/cracks", "crk", "valid")
    create_yaml()