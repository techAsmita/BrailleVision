import cv2
import streamlit as st
import numpy as np
from ultralytics import YOLO
import pyttsx3
import os
import threading

# Page config
st.set_page_config(
    page_title="BrailleVision",
    page_icon="👁️",
    layout="wide"
)

st.title("👁️ BrailleVision — Real-Time Braille Reader")
st.markdown("Point your camera at physical Braille text to convert it to English")

# Load model
MODEL_PATH = os.path.expanduser("~/braillevision/runs/detect/braille_detector-4/weights/best.pt")

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return YOLO(MODEL_PATH)
    else:
        st.error("Model not trained yet! Run train.py first.")
        return None

model = load_model()

# TTS function
def speak(text):
    def _speak():
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    thread = threading.Thread(target=_speak)
    thread.start()

# Camera guidance
def get_guidance(frame, detections):
    brightness = np.mean(frame)
    if brightness < 60:
        return "⚠️ Too dark — move to better lighting"
    elif brightness > 220:
        return "⚠️ Too bright — reduce glare"
    elif len(detections) == 0:
        return "📷 No Braille detected — move closer or adjust angle"
    elif len(detections) < 3:
        return "🔍 Few cells detected — try moving closer"
    else:
        return "✅ Braille detected!"

# Line by line reading
def extract_text_lines(detections, model):
    if detections is None or len(detections) == 0:
        return [], ""
    
    boxes_data = []
    for box in detections:
        x1 = float(box.xyxy[0][0])
        y1 = float(box.xyxy[0][1])
        cls = int(box.cls[0])
        label = model.names[cls]
        boxes_data.append((y1, x1, label))
    
    boxes_data.sort(key=lambda x: x[0])
    
    # Group into lines
    lines = []
    current_line = [boxes_data[0]]
    for box in boxes_data[1:]:
        if abs(box[0] - current_line[-1][0]) < 30:
            current_line.append(box)
        else:
            lines.append(sorted(current_line, key=lambda x: x[1]))
            current_line = [box]
    lines.append(sorted(current_line, key=lambda x: x[1]))
    
    letters = [b[2] for line in lines for b in line]
    text_lines = ["".join([b[2] for b in line]) for line in lines]
    full_text = "\n".join(text_lines)
    
    return letters, full_text

# Sidebar controls
st.sidebar.title("Controls")
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.4)
speak_output = st.sidebar.checkbox("🔊 Speak Output", value=True)
run_camera = st.sidebar.checkbox("📷 Start Camera", value=False)

# Main layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Camera Feed")
    frame_placeholder = st.empty()
    guidance_placeholder = st.empty()

with col2:
    st.subheader("Detected Text")
    text_placeholder = st.empty()
    st.subheader("Letters Found")
    letters_placeholder = st.empty()

# Camera loop
if run_camera and model:
    cap = cv2.VideoCapture(0)
    last_spoken = ""
    
    while run_camera:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera error!")
            break
        
        results = model(frame, conf=confidence, iou=0.3, verbose=False)
        detections = results[0].boxes
        
        guidance = get_guidance(frame, detections)
        guidance_placeholder.info(guidance)
        
        annotated = results[0].plot()
        letters, full_text = extract_text_lines(detections, model)
        
        text_placeholder.markdown(f"### `{full_text}`")
        letters_placeholder.markdown(
            " ".join([f"**{l}**" for l in letters]) if letters else "Waiting..."
        )
        
        if speak_output and full_text and full_text != last_spoken:
            speak(full_text)
            last_spoken = full_text
        
        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
    
    cap.release()

elif not run_camera:
    frame_placeholder.info("👆 Check 'Start Camera' in the sidebar to begin")

# Image upload
st.markdown("---")
st.subheader("📁 Or Upload a Braille Image")
uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded and model:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    results = model(img, conf=confidence, iou=0.3, verbose=False)
    detections = results[0].boxes
    annotated = results[0].plot()
    
    letters, full_text = extract_text_lines(detections, model)
    
    img_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="Detection Result", use_container_width=True)
    
    if full_text:
        st.success(f"**Detected Text:**")
        st.code(full_text)
        
        # Show line by line
        st.subheader("Line by Line:")
        for i, line in enumerate(full_text.split("\n")):
            st.write(f"Line {i+1}: **{line}**")
    else:
        st.warning("No Braille detected — try lowering confidence threshold")
    
    if speak_output and full_text:
        speak(full_text)