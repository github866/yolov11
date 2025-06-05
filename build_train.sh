#!/bin/bash

# Create root directory
mkdir -p train

# ReID directories
mkdir -p train/reid/dino_features
mkdir -p train/reid/configs
touch train/reid/index.faiss
touch train/reid/configs/reid.yaml

# Data directories
mkdir -p train/data/custom_dataset/images/train
mkdir -p train/data/custom_dataset/images/val
mkdir -p train/data/custom_dataset/images/test
mkdir -p train/data/custom_dataset/labels/train
mkdir -p train/data/custom_dataset/labels/val
mkdir -p train/data/custom_dataset/labels/test

# Dataset config
touch train/data/dataset.yaml

# Model directories
mkdir -p train/models/yolov11
mkdir -p train/models/pretrained
mkdir -p train/models/finetuned/runs

# Model config and weights
touch train/models/yolov11/yolov11.yaml
touch train/models/yolov11/yolov11_finetune.yaml
touch train/models/pretrained/yolov11s.pt

# Core training components
mkdir -p train/core
touch train/core/engine.py
touch train/core/validator.py
touch train/core/model.py

# Utils
mkdir -p train/utils
touch train/utils/dataloaders.py
touch train/utils/augmentations.py
touch train/utils/callbacks.py
touch train/utils/feature_extractors.py

# Configs
mkdir -p train/configs/experiments
touch train/configs/base.yaml
touch train/configs/finetune.yaml
touch train/configs/experiments/exp1_config.yaml
touch train/configs/experiments/exp2_config.yaml

# Outputs
mkdir -p train/outputs/logs
mkdir -p train/outputs/visualizations
mkdir -p train/outputs/metrics

# Experiments
mkdir -p train/experiments/exp1/weights
mkdir -p train/experiments/exp2
touch train/experiments/exp1/results.csv

# Scripts
mkdir -p train/scripts
touch train/scripts/start_training.sh
touch train/scripts/evaluate.sh

# Main scripts and requirements
touch train/train.py
touch train/detect.py
touch train/requirements.txt
touch train/README.md

echo "Project directory structure created."