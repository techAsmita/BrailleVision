from ultralytics import YOLO
import cv2
import os
import sys

# Load model
MODEL_PATH = os.path.expanduser("~/braillevision/runs/detect/braille_detector-4/weights/best.pt")
model = YOLO(MODEL_PATH)

def extract_text(image_path, conf=0.4, iou=0.3):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return ""
    
    results = model(img, conf=conf, iou=iou, verbose=False)
    detections = results[0].boxes
    
    if detections is None or len(detections) == 0:
        print("No Braille detected")
        return ""
    
    boxes_data = []
    for box in detections:
        x1 = float(box.xyxy[0][0])
        y1 = float(box.xyxy[0][1])
        cls = int(box.cls[0])
        label = model.names[cls]
        boxes_data.append((y1, x1, label))
    
    boxes_data.sort(key=lambda x: x[0])
    
    lines = []
    current_line = [boxes_data[0]]
    for box in boxes_data[1:]:
        if abs(box[0] - current_line[-1][0]) < 30:
            current_line.append(box)
        else:
            lines.append(sorted(current_line, key=lambda x: x[1]))
            current_line = [box]
    lines.append(sorted(current_line, key=lambda x: x[1]))
    
    text_lines = ["".join([b[2] for b in line]) for line in lines]
    full_text = "\n".join(text_lines)
    
    return full_text

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 inference.py <image_path>")
        print("Example: python3 inference.py sample_inputs/test_braille.jpg")
    else:
        image_path = sys.argv[1]
        print(f"Processing: {image_path}")
        result = extract_text(image_path)
        print(f"\nDetected Text:\n{result}")