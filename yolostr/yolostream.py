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
MODEL_PATH = "yolostr/cardmg.pt"  # Model path
try:
    model = YOLO(MODEL_PATH)
    st.success("Modèle chargé avec succès!")
except Exception as e:
    st.error(f"Erreur de chargement du modèle: {str(e)}")
    st.stop()

st.title("Détection de Dommages sur Véhicule")

def draw_detections(image, results, min_confidence=0.85):
    """Draw detection boxes on image with minimum confidence"""
    img_display = image.copy()
    detections = []
    
    for result in results:
        for box in result.boxes:
            conf = float(box.conf)
            if conf >= min_confidence:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                # Get class info
                cls_id = int(box.cls)
                class_name = CLASS_NAMES.get(cls_id, f"inconnu {cls_id}")
                
                # Draw rectangle
                cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
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
        # Read image
        image = Image.open(img_file).convert("RGB")
        img_array = np.array(image)
        
        # Perform detection with default confidence (0.25 to catch all possible detections)
        results = model.predict(
            source=img_array,
            conf=0.25,  # Low initial threshold to catch all possible detections
            imgsz=640,
            device='cpu'
        )
        
        # Draw and display results (filtering for ≥85% confidence)
        annotated_image, filtered_detections = draw_detections(img_array, results, min_confidence=0.85)
        st.image(annotated_image, caption="Résultats de détection (confiance ≥85%)", use_container_width=True)
        
        # Display filtered detections
        if filtered_detections:
            st.subheader("Dommages détectés (confiance ≥85%):")
            for det in sorted(filtered_detections, key=lambda x: x["confidence"], reverse=True):
                st.write(f"- {det['class_name']} (confiance: {det['confidence']:.2f})")
        else:
            st.warning("Aucun dommage détecté avec une confiance ≥85%")
            st.info("Suggestions:")
            st.write("- Essayez avec une image plus claire")
            st.write("- Prenez la photo sous un angle différent")
            st.write("- Vérifiez que les dommages sont visibles")
            
        # Debug info
        if st.checkbox("Afficher les informations de débogage"):
            st.write("Nombre total de détections:", len(results[0].boxes))
            st.write("Détections filtrées (≥85%):", len(filtered_detections))
            st.write("Exemple de résultat brut:", results[0].boxes[0] if len(results[0].boxes) > 0 else "Aucune")
            
    except Exception as e:
        st.error(f"Erreur de traitement: {str(e)}")
