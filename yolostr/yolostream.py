import streamlit as st
from PIL import Image
import tempfile
from ultralytics import YOLO
import cv2
import os

# Load the model (keep the model file in your GitHub repo)
model = YOLO("yolostr/cardmg.pt")  # Ensure this file exists in your repo

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
