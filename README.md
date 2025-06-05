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

train/                          # ROOT DIRECTORY
│
├── data/                       # All datasets
│   ├── custom_dataset/         # Your fine-tuning dataset
│   │   ├── images/             # Main image folder
│   │   │   ├── train/         # Training images
│   │   │   ├── val/           # Validation images
│   │   │   └── test/          # Test images
│   │   └── labels/            # Labels folder (mirrors images structure)
│   │       ├── train/         # Training labels
│   │       ├── val/           # Validation labels
│   │       └── test/          # Test labels
│   └── dataset.yaml            # Dataset config
│
├── models/                     # Model definitions & weights
│   ├── yolov11/                # Architecture configs
│   │   ├── yolov11.yaml        # Base config
│   │   └── yolov11_finetune.yaml # Fine-tuning adjustments
│   ├── pretrained/             # Pre-trained weights
│   │   └── yolov11s.pt         # Downloaded weights
│   └── finetuned/              # Output for trained weights
│       └── runs/               # Auto-saved checkpoints
│
├── core/                       # Main training components
│   ├── engine.py               # Training loop
│   ├── validator.py            # Validation logic
│   └── model.py                # Model initialization
│
├── utils/                      # Helper modules
│   ├── dataloaders.py          # Data loading/pipeline
│   ├── augmentations.py        # Custom image augmentations
│   ├── callbacks.py            # Training callbacks
│   └── feature_extractors.py   # Feature extraction utils
│
├── configs/                    # Training configurations
│   ├── base.yaml               # Default hyperparameters
│   ├── finetune.yaml           # Fine-tuning specific config
│   └── experiments/            # Per-run configs
│       ├── exp1_config.yaml
│       └── exp2_config.yaml
│
├── outputs/                    # Training artifacts
│   ├── logs/                   # TensorBoard/CSV logs
│   ├── visualizations/         # Feature maps, attention, etc.
│   └── metrics/                # Detailed metrics
│
├── experiments/                # Experiment tracking
│   ├── exp1/                   # Experiment 1
│   │   ├── weights/            # Best checkpoint
│   │   └── results.csv         # Performance metrics
│   └── exp2/                   # Experiment 2
│
├── scripts/                    # Automation scripts
│   ├── start_training.sh       # Launch training
│   └── evaluate.sh             # Run evaluation
│
├── train.py                    # MAIN TRAINING SCRIPT
├── detect.py                   # Inference script
├── requirements.txt            # Dependencies
└── README.md                   # Project documentation