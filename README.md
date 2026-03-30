# Text-Guided Segmentation

This repository contains a complete pipeline for text-conditioned segmentation, specifically targeting drywall joints and surface cracks. It implements a progression of architectures from zero-shot foundation models to a highly optimized, domain-specific cascaded pipeline (Fine-tuned YOLO-World + Custom YOLO-seg).

## Table of Contents
- [Architecture Overview](#architecture-overview)
- [Setup and Installation](#setup-and-installation)
- [Dataset Preparation](#dataset-preparation)
- [Training Pipeline](#training-pipeline)
- [Evaluation and Inference](#evaluation-and-inference)
- [Repository Structure](#repository-structure)
<!-- - [Utilities](#utilities)
- [Results Summary](#results-summary) -->

## Architecture Overview
The optimal solution in this repository utilizes a **Text-Guided Cascade**:
1. **Detection Stage (YOLO-World):** Takes a natural language prompt (e.g., "segment crack") and an image, outputting precise bounding boxes for the requested defect.
2. **Segmentation Stage (YOLO-seg):** A custom-trained segmentation model processes the image. The bounding boxes from Stage 1 act as a strict spatial filter, extracting only the pixel masks that align with the initial text query.

## Setup and Installation

Ensure you have a CUDA-capable environment (CUDA 11.8+ recommended).

```bash
# Install PyTorch
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# Install required packages
pip install ultralytics opencv-python numpy pycocotools tqdm pyyaml

# Install Segment Anything (for baseline comparisons)
pip install git+[https://github.com/facebookresearch/segment-anything.git](https://github.com/facebookresearch/segment-anything.git)

# Download the pre-trained SAM weights to the weights/ directory for baseline evaluation:
mkdir weights
wget -O weights/sam_vit_h_4b8939.pth [https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
```
## Downloading Pretrained Weights for YOLO-World and YOLO-Seg Models

To download the pretrained weights for the YOLO-World and YOLO-Seg models, follow these instructions:

1. **YOLO-World**:
   - You can download the pretrained weights from the following link: [YOLO-World Weights](https://drive.google.com/file/d/10MunvBRf-NF0GP0m6DhJ9_1yhgHkwhG_/view?usp=sharing).
   - Download path: runs/detect/yolo_world_combined_v1/weights

2. **YOLO-Seg**:
   - Download the pretrained weights using this link: [YOLO-Seg Weights](https://drive.google.com/file/d/1g7eSFrWTejZ13X-jnALa7IOo1spa24-l/view?usp=sharing)
   - Download path: runs/segment/runs/segment/yolo_seg_combined/weights

Place the downloaded weights in the appropriate directory as defined in the project documentation for seamless integration.

## Dataset Preparation
The original datasets are in COCO JSON format. To train the YOLO models, convert the segmentation masks and bounding boxes into the YOLO format.

Place your raw datasets in data/drywall_join and data/cracks.

Run the conversion script to generate a unified YOLO-ready dataset at data/yolo_seg_dataset/ and create the data.yaml file:

```bash
python -m utils.convert_data_combined
python -m utils.convert_yolo_seg
```
## Training Pipeline
The cascaded pipeline requires domain-specific fine-tuning for both stages to overcome out-of-the-box localization failures and annotator bias.

1. Train YOLO-World (Detector)
Fine-tune the open-vocabulary detector to accurately localize defects based on text prompts.

```Bash
python -m pipelines.train_yolo
```
2. Train YOLO-seg (Segmenter)
Train the dedicated segmentation model on the full-resolution defect dataset.

```Bash
python -m pipelines.train_yolo_seg
```

## Evaluation and Inference
The evaluation script processes the validation sets, calculates mIoU and mDice scores, applies necessary morphological dilations (to counter GT annotator bias), saves binary rubric masks, and exports detailed CSV logs.

To run the full evaluation across all validation datasets:

```Bash
python -m evaluation.evaluate_trained_yoloworld
```
Outputs will be generated in:

evaluation/rubric_masks/: Visual PNG masks for the rubric.

evaluation/detailed_results.csv: Image-by-image metric breakdown.

To infer on a particuar image:
```Bash
python -m pipelines.main
```

## Repository Structure

### Directory Descriptions:
- **models/**: Contains detector and segmenter modules with model implementations
- **pipelines/**: Training pipelines for YOLO-World, YOLO-Seg, and DINO models
- **evaluation/**: Evaluation scripts, metrics computation, and result visualizations
- **utils/**: Data conversion, visualization, and utility functions
- **runs/**: Generated model checkpoints and training artifacts
- **data/**: Raw and processed datasets in multiple formats
- **weights/**: Pre-trained model weights
|── runs/ 
|   ├── detector
│   ├── segmenter
└── README.md
