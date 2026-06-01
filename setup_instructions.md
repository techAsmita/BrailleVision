# Setup Instructions

## Requirements
- Python 3.8+
- Mac/Windows/Linux

## Installation
```bash
git clone https://github.com/techAsmita/BrailleVision.git
cd BrailleVision
pip install -r requirements.txt
```

## Run the App
```bash
python3 -m streamlit run app.py
```

## Run Inference on Image
```bash
python3 inference.py sample_inputs/your_image.jpg
```

## Model
Model weights are in:
runs/detect/braille_detector-4/weights/best.pt

## Dataset
See Dataset_info.md for details