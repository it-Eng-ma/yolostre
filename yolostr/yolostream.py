import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO

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

# Load PyTorch model
MODEL_PATH = "cardmg.pt"  # Ensure this is in your root directory
try:
    model = YOLO(MODEL_PATH)
    st.success("Modèle chargé avec succès!")
except Exception as e:
    st.error(f"Erreur de chargement du modèle: {str(e)}")
    st.stop()

st.title("Détection de Dommages sur Véhicule")

def draw_detections(image, results):
    """Draw detection boxes on image"""
    img_display = image.copy()
    
    for result in results:
        for box in result.boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # Get class and confidence
            cls_id = int(box.cls)
            conf = float(box.conf)
            class_name = CLASS_NAMES.get(cls_id, f"inconnu {cls_id}")
            
            # Draw rectangle
            cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name} {conf:.2f}"
            cv2.putText(img_display, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return img_display

# File uploader with confidence threshold
img_file = st.file_uploader("Télécharger une image de véhicule", type=["jpg", "jpeg", "png"])
conf_threshold = st.slider("Seuil de confiance", 0.01, 1.0, 0.25, 0.01)

if img_file:
    try:
        # Read image
        image = Image.open(img_file).convert("RGB")
        img_array = np.array(image)
        
        # Perform detection
        results = model.predict(
            source=img_array,
            conf=conf_threshold,
            imgsz=640,
            device='cpu'  # Use 'cuda' if GPU available
        )
        
        # Draw and display results
        annotated_image = draw_detections(img_array, results)
        st.image(annotated_image, caption="Résultats de détection", use_container_width=True)
        
        # List detections
        detections = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls)
                detections.append({
                    "class_name": CLASS_NAMES.get(cls_id, f"inconnu {cls_id}"),
                    "confidence": float(box.conf),
                    "coords": box.xyxy[0].tolist()
                })
        
        if detections:
            st.subheader("Dommages détectés:")
            for det in sorted(detections, key=lambda x: x["confidence"], reverse=True):
                st.write(f"- {det['class_name']} (confiance: {det['confidence']:.2f})")
        else:
            st.warning("Aucun dommage détecté - Essayez avec:")
            st.write("- Une image plus claire")
            st.write("- Un angle différent")
            st.write("- Un seuil de confiance plus bas")
            
        # Debug info
        if st.checkbox("Afficher les informations de débogage"):
            st.write("Résultats bruts:", results[0].boxes)
            st.write("Nombre de détections:", len(detections))
            
    except Exception as e:
        st.error(f"Erreur de traitement: {str(e)}")
