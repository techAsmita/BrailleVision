from ultralytics import YOLO
import os

# Fix dataset yaml paths
dataset_path = os.path.expanduser("~/braillevision/dataset")
yaml_path = os.path.join(dataset_path, "data.yaml")

# Update yaml with absolute paths
with open(yaml_path, 'r') as f:
    content = f.read()

content = content.replace('../train/images', f'{dataset_path}/train/images')
content = content.replace('../valid/images', f'{dataset_path}/valid/images')
content = content.replace('../test/images', f'{dataset_path}/test/images')

with open(yaml_path, 'w') as f:
    f.write(content)

print("Dataset paths fixed!")

# Load YOLOv8 nano model and train
model = YOLO('yolov8n.pt')

model.train(
    data=yaml_path,
    epochs=100,       
    imgsz=640,       
    batch=8,
    name='braille_detector',
    patience=10,
    save=True,
    plots=True
)

print("Training complete! Model saved.")