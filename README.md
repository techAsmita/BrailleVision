# BrailleVision 👁️
### Real-Time Physical Braille to English Converter
**BrailleVision Hackathon 2026 Submission**

## What It Does
Detects real physical embossed Braille from a camera feed and converts it to English text and speech in real time.

## Pipeline
Camera → YOLOv8 Detection → Letter Recognition → English Text → Text-to-Speech

## Tech Stack
- Python
- YOLOv8 (Ultralytics)
- OpenCV
- Streamlit
- pyttsx3 (TTS)
- Roboflow Dataset

## How to Run
```bash
pip install -r requirements.txt
python3 -m streamlit run app.py
```

## Dataset
- Source: Roboflow Universe
- 1016 training images
- 26 classes (A-Z)
- Format: YOLOv8

## Model
- Base: YOLOv8 nano
- Fine-tuned on physical Braille dataset
- Output: best.pt

## AI Tools Used
See ai_tools_disclosure.md