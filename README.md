# Room Segmentation and Person Localization

This project provides tools to segment rooms in camera feeds and determine which room a person is in, particularly useful for tracking people near boundaries between rooms.

## Setup

Install required dependencies:

```bash
pip install opencv-python numpy torch ultralytics shapely
```

## Usage

The workflow consists of three main steps:

### 1. Room Segmentation

First, define the room boundaries using the segmentation tool:

```bash
python room_segmentation.py
```

This will prompt you to:
1. Enter a path to a reference image
2. Draw room boundaries by clicking and dragging
3. Press 'n' to name each room after drawing its boundary
4. Press 's' to save the boundaries to a JSON file

### 2. Test with Existing YOLO Results

If you already have YOLO detection results, you can test the room localization:

```bash
python room_localization.py --boundaries rooms.json --image image.jpg --yolo_results detections.json --output visualization.jpg
```

### 3. Real-time Detection and Localization

For real-time detection with YOLOv11:

```bash
nohup python human_tracking.py --input_folder /data/zhaoheng_zhu/origin/Camera-Loc04 --output_excel Loc04.xlsx --output_images output_visualizations_loc04 --json_file loc04.json     --output_json detections_loc04.json --output_yaml train-loc04.yaml > track4.log 2>&1 &
```

For webcam input:

```bash
python yolo_room_detection.py --model yolov11n.pt --boundaries rooms.json --source 0
```

## Room Boundary Definition

When defining room boundaries:
- Draw complete polygons around each room
- Make sure boundaries don't overlap
- For doorways or openings between rooms, draw the boundary line at the threshold

## Localization Rules

The system uses these rules to determine a person's location:
1. A person is considered in a room if their feet (bottom center of bounding box) are inside the room boundary
2. If a person is near a boundary (within 20 pixels), they're marked as "near boundary"
3. If a person is not in any defined room, they're marked as "Unknown"

## Customization

The localization rules can be adjusted in `room_localization.py`:
- Modify the distance threshold (20 pixels) for boundary detection
- Change the point used for localization (currently bottom center of bounding box)
- Adjust the visualization style

## Requirements

- Python 3.6+
- OpenCV
- NumPy
- PyTorch
- Ultralytics YOLOv11
- Shapely

# Project Directory Structure for Training and Data Management

root/
├── train/                  # Main training directory (your suggestion - good!)
│   ├── data/               # Custom dataset in YOLO format
│   │   ├── images/         # All images
│   │   │   ├── train/      # Training images
│   │   │   └── val/        # Validation images
│   │   └── labels/         # Corresponding labels
│   │       ├── train/      # Training labels
│   │       └── val/        # Validation labels
│   │
│   ├── config/             # Configuration files
│   │   ├── dataset.yaml    # Dataset config (paths, class names)
│   │   └── hyperparameters.yaml  # Custom hyperparameters
│   │
│   ├── weights/            # Model weights
│   │   ├── pretrained/     # Official pretrained weights
│   │   └── finetuned/      # Fine-tuned checkpoints (output)
│   │
│   ├── runs/               # Training outputs (auto-created by YOLO)
│   │   ├── train/          # Training logs, metrics, predictions
│   │   └── val/            # Validation results
│   │
│   ├── scripts/            # Custom utilities
│   │   ├── preprocess.py   # Data preprocessing
│   │   └── visualize.py    # Result visualization
│   │
│   └── train.py            # Main training script
│
├── models/                 # Custom model definitions (optional)
│   └── yolov11-custom.yaml # Modified architecture
│
├── docs/                   # Documentation
└── README.md
