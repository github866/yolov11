from core.engine import Trainer
from models.yolov11 import YOLOv11

# Load configuration
cfg = load_config('./configs/finetune.yaml')

# Initialize model
model = YOLOv11(
    cfg='./models/yolov11/yolov11s.yaml',
    weights='./models/pretrained/yolov11s.pt'
)

# Start training
trainer = Trainer(
    model=model,
    data_cfg='./data/dataset.yaml',
    output_dir='./outputs/'
)
trainer.fit()