import streamlit as st
from PIL import Image
import tempfile
from ultralytics import YOLO
import cv2
import os
import numpy as np
import requests

# Define model path and download URL
model_path = "yolostr/cardmg.pt"
model_url = "https://github.com/it-Eng-ma/yolostre/raw/main/yolostr/cardmg.pt"

# Download the model if it doesn't exist
if not os.path.exists(model_path):
    os.makedirs("yolostr", exist_ok=True)
    with st.spinner("Downloading model..."):
        response = requests.get(model_url)
        if response.status_code == 200:
            with open(model_path, "wb") as f:
                f.write(response.content)
        else:
            st.error(f"Failed to download model. HTTP status code: {response.status_code}")
            st.stop()

# Load the model
model = YOLO(model_path)

st.title("Car Damage Detection")

# Upload image
img_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if img_file is not None:
    # Use PIL to read image (better compatibility)
    image = Image.open(img_file)
    image_np = np.array(image)  # Convert to numpy array for OpenCV

    # Temporary file handling (optional - you can use the array directly)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        tmp_path = tmp.name

    # Perform inference
    results = model(image_np)  # Directly use numpy array

    # Process results
    if len(results) > 0:
        res_plotted = results[0].plot()  # Get annotated image
        
        # Convert BGR to RGB for Streamlit display
        res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        st.image(res_plotted_rgb, caption="Detected Damages", use_column_width=True)

        # Show detected classes
        class_names = results[0].names
        boxes = results[0].boxes
        detected_classes = set()
        
        if boxes is not None and boxes.cls is not None:
            for box in boxes.cls:
                detected_classes.add(class_names[int(box)])
            
            st.subheader("Detected Damage Parts:")
            for cls in detected_classes:
                st.write(f"- {cls}")
        else:
            st.write("No damages detected")
    else:
        st.write("No detection results")

    # Clean up temp file
    os.unlink(tmp_path)



