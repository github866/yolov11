# Image Cropper Tool

A graphical tool for interactively cropping regions from a sequence of images, designed for computer vision tasks such as object detection and tracking.

## Features

- Interactive GUI for selecting and cropping regions from image sequences
- Support for multiple crop regions per image
- Identity labeling system for annotating crops (e.g., nurse, patient, etc.)
- Automatic saving of crop coordinates in JSON format
- Navigation controls for moving between frames
- Resizable crop regions with handle controls
- Support for both original and processed image directories

## Usage

### Basic Usage

```bash
python3 cropper.py --image_dir <image_directory> --json_name <output_json_name>
```

### Example Commands

For processing YOLO results:
```bash
python3 cropper.py --image_dir yolo_results/clip4/ --json_name clip4_missing.json
```

For processing original images:
```bash
python3 cropper.py --image_dir clip3/ --json_name clip3_missing.json
```

### Command Line Arguments

- `--image_dir`: Directory containing the image sequence (default: "/")
- `--output_dir`: Directory for saving cropped images (default: "cropped_images")
- `--json_name`: Name of the output JSON file (default: "clip1_missing.json")
- `--current_subject_index`: Initial subject index for identity selection (default: 0)

## Output Format

The tool saves crop data in a JSON file with the following structure:

```json
[
  {
    "frame": 1,
    "persons": [
      {
        "subject_id": 0,
        "subject_name": "nurse_1",
        "coordinate": [x1, y1, x2, y2]
      }
    ]
  }
]
```

## Supported Identity Labels

- nurse_1, nurse_2
- patient_1, patient_2, patient_3
- psychiatrist
- psychologist
- researcher
- person_1, person_2, person_3, person_4

## Controls

- Left/Right arrow keys: Navigate between frames
- Mouse: Draw and adjust crop regions
- Save Crop button: Save current crop coordinates
- Previous/Next buttons: Navigate between frames
- Frame selection: Directly input frame number
- Identity dropdown: Select identity label for current crop