import streamlit as st
from PIL import Image
import tempfile
import cv2
import numpy as np
import os
import onnxruntime as ort

# Set model path
model_path = "yolostr/cardmg.onnx"

# Load ONNX model
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

st.title("Car Damage Detection")

# Upload image
img_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

def preprocess_image(image, target_size=640):
    # Resize and pad image to square
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (nw, nh))
    padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    padded[:nh, :nw] = resized

    # Normalize to 0-1 and transpose to (1, 3, 640, 640)
    img = padded.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[np.newaxis, :]
    return img, scale, (nw, nh)

def postprocess(outputs, scale, size, conf_thresh=0.3, iou_thresh=0.5):
    # YOLOv8 ONNX output format: [batch, num_preds, 85] — [x, y, w, h, conf, class scores...]
    preds = outputs[0][0]  # remove batch dim
    boxes = []
    for pred in preds:
        conf = pred[4]
        cls_conf = np.max(pred[5:])
        cls_id = np.argmax(pred[5:])
        if conf * cls_conf > conf_thresh:
            cx, cy, w, h = pred[:4]
            x1 = int((cx - w / 2) * size[0] / scale)
            y1 = int((cy - h / 2) * size[1] / scale)
            x2 = int((cx + w / 2) * size[0] / scale)
            y2 = int((cy + h / 2) * size[1] / scale)
            boxes.append((x1, y1, x2, y2, int(cls_id), float(conf * cls_conf)))
    return boxes

if img_file is not None:
    # Read and convert image
    image = Image.open(img_file).convert("RGB")
    image_np = np.array(image)

    # Preprocess image
    input_tensor, scale, size = preprocess_image(image_np)

    # Run inference
    outputs = session.run([output_name], {input_name: input_tensor})

    # Postprocess predictions
    boxes = postprocess(outputs, scale, size)

    # Annotate image
    for x1, y1, x2, y2, cls_id, score in boxes:
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Damage {cls_id} ({score:.2f})"
        cv2.putText(image_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Display result
    st.image(image_np, caption="Detected Damages", use_column_width=True)

    if boxes:
        st.subheader("Detected Damage Boxes:")
        for x1, y1, x2, y2, cls_id, score in boxes:
            st.write(f"- Class {cls_id} at [{x1}, {y1}, {x2}, {y2}] with confidence {score:.2f}")
    else:
        st.write("No damages detected.")
