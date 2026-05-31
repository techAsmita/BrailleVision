# AI Tools Disclosure

## AI Tools Used
- Claude (Anthropic) — used for code assistance and project guidance
- Roboflow — used for dataset sourcing
- Ultralytics YOLOv8 — pre-trained model used as base

## What Was Built During Hackathon
- Complete training pipeline (train.py)
- Real-time Streamlit application (app.py)
- YOLO inference integration
- Text-to-speech output
- Camera guidance system
- Image upload fallback

## What Was Reused
- YOLOv8 nano pretrained weights (yolov8n.pt) from Ultralytics
- Braille Detection dataset from Roboflow Universe (CC BY 4.0)
- Source: https://universe.roboflow.com/braille-lq5eh/braille-detection

## What Was Modified
- Dataset paths configured for local training
- Model fine-tuned on Braille detection dataset