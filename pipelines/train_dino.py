import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from groundingdino.util.inference import load_model
from groundingdino.models import build_model
from groundingdino.util.misc import collate_fn
from groundingdino.datasets.transforms import make_coco_transforms

# =========================

# CONFIG

# =========================

DEVICE = "cuda"
BATCH_SIZE = 2
LR = 1e-5
EPOCHS = 10

DATA_ROOT = "data"

PROMPT = "crack, drywall joint, wall damage"

MODEL_CONFIG = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
MODEL_WEIGHTS = "weights/groundingdino_swint_ogc.pth"

SAVE_PATH = "checkpoints/dino_finetuned.pth"


class GroundingCocoDataset(CocoDetection):
    def __init__(self, img_folder, ann_file, transforms=None, prompt=""):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prompt = prompt

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)

        # Convert to DINO format
        boxes = []
        labels = []

        for obj in target:
            bbox = obj["bbox"]  # COCO format [x, y, w, h]
            x, y, w, h = bbox
            boxes.append([x, y, x + w, y + h])
            labels.append(obj["category_id"])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "caption": self.prompt  # TEXT CONDITIONING
        }

        if self._transforms:
            img, target = self._transforms(img, target)

        return img, target



def build_dataloader(dataset_name):
    train_path = os.path.join(DATA_ROOT, dataset_name, "train")
    val_path = os.path.join(DATA_ROOT, dataset_name, "valid")


    train_dataset = GroundingCocoDataset(
        img_folder=train_path,
        ann_file=os.path.join(train_path, "annotations.json"),
        transforms=make_coco_transforms("train"),
        prompt=PROMPT
    )

    val_dataset = GroundingCocoDataset(
        img_folder=val_path,
        ann_file=os.path.join(val_path, "annotations.json"),
        transforms=make_coco_transforms("val"),
        prompt=PROMPT
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )

    return train_loader, val_loader


def train():
    model = build_model(MODEL_CONFIG)
    checkpoint = torch.load(MODEL_WEIGHTS, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)


    model.to(DEVICE)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # Combine both datasets
    datasets = ["cracks", "drywall_join"]
    loaders = [build_dataloader(ds)[0] for ds in datasets]

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch}")

        for loader in loaders:
            for images, targets in loader:
                images = [img.to(DEVICE) for img in images]
                targets = [{k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]

                outputs = model(images, targets)

                loss_dict = outputs["loss_dict"]
                loss = sum(loss_dict.values())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print(f"Loss: {loss.item()}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), SAVE_PATH)


if __name__ == "__main__":
    train()
