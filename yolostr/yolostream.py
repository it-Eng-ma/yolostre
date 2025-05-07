import streamlit as st
from PIL import Image
import cv2
import numpy as np
import os
import onnxruntime as ort

# Configuration - French damage labels
CLASS_NAMES = {
    0: "porte endommagée",
    1: "fenêtre endommagée", 
    2: "phare endommagé",
    3: "rétroviseur endommagé",
    4: "bosse",
    5: "capot endommagé",
    6: "pare-chocs endommagé",
    7: "pare-brise endommagé"
}

# Load ONNX model
MODEL_PATH = "yolostr/cardmg.onnx"
if not os.path.exists(MODEL_PATH):
    st.error(f"Modèle introuvable: {MODEL_PATH}")
    st.stop()

try:
    session = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name  # Properly define output_name
except Exception as e:
    st.error(f"Erreur de chargement du modèle: {str(e)}")
    st.stop()

st.title("Détection de Dommages sur Véhicule")

def preprocess_image(image, target_size=640):
    """Convert and normalize image for YOLO model"""
    img_array = np.array(image)
    if len(img_array.shape) == 2:  # Grayscale
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    else:  # RGB
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    h, w = img_array.shape[:2]
    scale = min(target_size / h, target_size / w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img_array, (nw, nh))
    
    padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    padded[:nh, :nw] = resized
    
    img = padded.astype(np.float32) / 255.0
    return np.transpose(img, (2, 0, 1))[np.newaxis, :], scale, (w, h)

def postprocess(outputs, scale, original_size, conf_thresh=0.25):
    """Process YOLOv8 ONNX model output"""
    try:
        predictions = outputs[0][0]  # Shape: [84, 8400]
        boxes = []
        
        for pred in predictions.T:  # [8400, 84]
            *xywh, conf, class_scores = np.split(pred, [4, 5, 5 + len(CLASS_NAMES)])
            cls_id = np.argmax(class_scores)
            total_conf = conf * class_scores[cls_id]
            
            if total_conf > conf_thresh:
                x_center, y_center, w, h = xywh
                x1 = int((x_center - w/2) * original_size[0] / scale)
                y1 = int((y_center - h/2) * original_size[1] / scale)
                x2 = int((x_center + w/2) * original_size[0] / scale)
                y2 = int((y_center + h/2) * original_size[1] / scale)
                
                boxes.append({
                    "coords": [max(0, x1), max(0, y1), 
                              min(original_size[0], x2), 
                              min(original_size[1], y2)],
                    "class_id": int(cls_id),
                    "confidence": float(total_conf),
                    "class_name": CLASS_NAMES.get(int(cls_id), f"inconnu {cls_id}")
                })
        
        return boxes
    except Exception as e:
        st.error(f"Erreur d'analyse: {str(e)}")
        return []

# File uploader
img_file = st.file_uploader("Télécharger une image de véhicule", type=["jpg", "jpeg", "png"])

if img_file:
    try:
        image = Image.open(img_file).convert("RGB")
        original_w, original_h = image.size
        
        # Preprocess
        input_tensor, scale, original_size = preprocess_image(image)
        
        # Inference - now using properly defined output_name
        outputs = session.run([output_name], {input_name: input_tensor})
        
        # Postprocess
        detections = postprocess(outputs, scale, (original_w, original_h))
        
        # Draw results
        img_display = np.array(image)
        for det in detections:
            x1, y1, x2, y2 = det["coords"]
            cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.putText(img_display, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        st.image(img_display, caption="Résultats de détection", use_column_width=True)
        
        if detections:
            st.subheader("Dommages détectés:")
            for det in sorted(detections, key=lambda x: x["confidence"], reverse=True):
                st.write(f"- {det['class_name']} (confiance: {det['confidence']:.2f})")
        else:
            st.warning("Aucun dommage détecté")
            
    except Exception as e:
        st.error(f"Erreur de traitement: {str(e)}")
