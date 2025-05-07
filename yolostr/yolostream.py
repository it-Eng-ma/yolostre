import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO

# Configuration - French damage labels
CLASS_NAMES = {
    0: "porte endommagee",
    1: "fenetre endommagee", 
    2: "phare endommage",
    3: "retroviseur endommage",
    4: "bosse",
    5: "capot endommage",
    6: "pare-chocs endommage",
    7: "pare-brise endommage"
}

# Load PyTorch model
MODEL_PATH = "yolostr/cardmg.pt"
try:
    model = YOLO(MODEL_PATH)
    st.success("Modèle chargé avec succès!")
except Exception as e:
    st.error(f"Erreur de chargement du modèle: {str(e)}")
    st.stop()

st.title("Détection de Dommages sur Véhicule")

def draw_detections(image, results):
    """Draw detection boxes with ≥85% confidence"""
    img_display = image.copy()
    detections = []
    
    for result in results:
        for box in result.boxes:
            conf = float(box.conf)
            if conf >= 0.85:  # Hardcoded 85% confidence threshold
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls)
                class_name = CLASS_NAMES.get(cls_id, f"inconnu {cls_id}")
                
                # Draw rectangle and label
                cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name} {conf:.2f}"
                cv2.putText(img_display, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                detections.append({
                    "class_name": class_name,
                    "confidence": conf,
                    "coords": [x1, y1, x2, y2]
                })
    
    return img_display, detections

# File uploader
img_file = st.file_uploader("Télécharger une image de véhicule", type=["jpg", "jpeg", "png"])

if img_file:
    try:
        # Read and process image
        image = Image.open(img_file).convert("RGB")
        img_array = np.array(image)
        
        # Perform detection (low initial threshold to catch all possibilities)
        results = model.predict(
            source=img_array,
            conf=0.25,
            imgsz=640,
            device='cpu'
        )
        
        # Get filtered results (≥85% confidence only)
        annotated_image, filtered_detections = draw_detections(img_array, results)
        st.image(annotated_image, caption="Dommages détectés (confiance ≥85%)", use_container_width=True)
        
        # Display results
        if filtered_detections:
            st.subheader("Dommages confirmés:")
            for det in sorted(filtered_detections, key=lambda x: x["confidence"], reverse=True):
                st.write(f"- {det['class_name']} (certitude: {det['confidence']:.0%}")
        else:
            st.warning("Aucun dommage significatif détecté")
            st.info("Conseils pour une meilleure détection:")
            st.write("• Photographiez sous un angle direct")
            st.write("• Assurez un bon éclairage")
            st.write("• Capturez les détails de près")
            
        # Optional debug info
        if st.checkbox("Afficher les détails techniques"):
            st.write("Total des détections potentielles:", len(results[0].boxes))
            st.write("Détections validées (≥85%):", len(filtered_detections))
            
    except Exception as e:
        st.error(f"Erreur lors de l'analyse: {str(e)}")
