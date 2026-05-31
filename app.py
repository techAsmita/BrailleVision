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
MODEL_PATH = os.path.expanduser("~/braillevision/runs/detect/braille_detector-3/weights/best.pt")

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
    h, w = frame.shape[:2]
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

# Sidebar controls
st.sidebar.title("Controls")
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)
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
        
        # Run YOLO detection
        results = model(frame, conf=confidence, verbose=False)
        detections = results[0].boxes
        
        # Get guidance message
        guidance = get_guidance(frame, detections)
        guidance_placeholder.info(guidance)
        
        # Draw detections and collect letters
        annotated = results[0].plot()
        letters = []
        
        if detections is not None and len(detections) > 0:
            # Sort by x position (left to right reading order)
            boxes_data = []
            for box in detections:
                x1 = float(box.xyxy[0][0])
                conf_score = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls]
                boxes_data.append((x1, label, conf_score))
            
            boxes_data.sort(key=lambda x: x[0])
            letters = [item[1] for item in boxes_data]
        
        # Display results
        detected_text = " ".join(letters) if letters else "..."
        text_placeholder.markdown(f"### `{detected_text}`")
        letters_placeholder.markdown(
            " ".join([f"**{l}**" for l in letters]) if letters else "Waiting..."
        )
        
        # Speak if new text
        if speak_output and detected_text != "..." and detected_text != last_spoken:
            speak(detected_text)
            last_spoken = detected_text
        
        # Show frame
        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
    
    cap.release()

elif not run_camera:
    frame_placeholder.info("👆 Check 'Start Camera' in the sidebar to begin")

# Image upload option
st.markdown("---")
st.subheader("📁 Or Upload a Braille Image")
uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded and model:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    results = model(img, conf=confidence, verbose=False)
    detections = results[0].boxes
    annotated = results[0].plot()
    
    letters = []
    if detections is not None and len(detections) > 0:
        boxes_data = []
        for box in detections:
            x1 = float(box.xyxy[0][0])
            cls = int(box.cls[0])
            label = model.names[cls]
            boxes_data.append((x1, label))
        boxes_data.sort(key=lambda x: x[0])
        letters = [item[1] for item in boxes_data]
    
    img_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="Detection Result", use_container_width=True)
    
    detected_text = "".join(letters)
    st.success(f"Detected: **{detected_text}**")
    
    if speak_output and detected_text:
        speak(detected_text)