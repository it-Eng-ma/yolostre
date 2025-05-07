import streamlit as st
from PIL import Image
import tempfile
import cv2
import numpy as np
import os
import onnxruntime as ort

# Configuration
MODEL_PATH = "yolostr/cardmg.onnx"
CLASS_NAMES = {
    0: "Front Damage",
    1: "Rear Damage",
    2: "Side Damage",
    3: "Window Crack"
}

# Load ONNX model
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name

st.title("Car Damage Detection")

def preprocess_image(image, target_size=640):
    # Convert PIL to OpenCV format
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Resize maintaining aspect ratio
    h, w = image.shape[:2]
    scale = min(target_size / h, target_size / w)
    nh, nw = int(h * scale), int(w * scale)
    
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    
    # Pad to square
    padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    padded[:nh, :nw] = resized
    
    # Normalize and transpose
    img = padded.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)[np.newaxis, :]  # (1, 3, 640, 640)
    
    return img, scale, (nw, nh)

def postprocess(outputs, scale, original_size, conf_thresh=0.3):
    # Assuming model output is (1, 84, 8400)
    preds = outputs[0][0]  # Remove batch dimension
    boxes = []
    
    # Get number of classes from CLASS_NAMES
    num_classes = len(CLASS_NAMES)
    
    # Split predictions
    for pred in preds.T:  # Transpose to (8400, 84)
        *xywh, conf, class_scores = np.split(pred, [4, 5, 5+num_classes])
        cls_id = np.argmax(class_scores)
        cls_conf = np.max(class_scores)
        
        if conf * cls_conf > conf_thresh:
            x_center, y_center, w, h = xywh
            x1 = int((x_center - w/2) / scale)
            y1 = int((y_center - h/2) / scale)
            x2 = int((x_center + w/2) / scale)
            y2 = int((y_center + h/2) / scale)
            
            # Clip coordinates to image dimensions
            x1 = max(0, min(x1, original_size[0]))
            y1 = max(0, min(y1, original_size[1]))
            x2 = max(0, min(x2, original_size[0]))
            y2 = max(0, min(y2, original_size[1]))
            
            boxes.append((x1, y1, x2, y2, cls_id, float(conf * cls_conf)))
    
    return boxes

# File uploader
img_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if img_file is not None:
    try:
        # Read and preprocess
        image = Image.open(img_file)
        input_tensor, scale, size = preprocess_image(image)
        original_w, original_h = image.size
        
        # Inference
        outputs = session.run(None, {input_name: input_tensor})
        
        # Postprocess
        boxes = postprocess(outputs, scale, (original_w, original_h))
        
        # Draw results
        image_np = np.array(image)
        for x1, y1, x2, y2, cls_id, score in boxes:
            # Draw rectangle
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Get class name
            class_name = CLASS_NAMES.get(cls_id, f"Unknown {cls_id}")
            
            # Draw label
            label = f"{class_name} {score:.2f}"
            cv2.putText(image_np, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Display
        st.image(image_np, caption="Detected Damages", use_column_width=True)
        
        # Show detections
        if boxes:
            st.subheader("Detected Damages:")
            for x1, y1, x2, y2, cls_id, score in boxes:
                class_name = CLASS_NAMES.get(cls_id, f"Unknown {cls_id}")
                st.write(f"- {class_name} (Confidence: {score:.2f})")
        else:
            st.write("No damages detected.")
            
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
