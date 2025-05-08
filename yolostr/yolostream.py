import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
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
    """Draw detection boxes with ≥85% confidence using PIL"""
    img_display = image.copy()
    draw = ImageDraw.Draw(img_display)
    
    # Use a basic font (size=15 as a starting point)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except:
        font = ImageFont.load_default()  # Fallback if font not found
    
    detections = []
    
    for result in results:
        for box in result.boxes:
            conf = float(box.conf)
            if conf >= 0.85:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls)
                class_name = CLASS_NAMES.get(cls_id, f"inconnu {cls_id}")
                
                # Draw rectangle and label (PIL uses RGB colors)
                draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
                label = f"{class_name} {conf:.2f}"
                draw.text((x1, y1 - 15), label, fill="red", font=font)
                
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
        # Read image directly with PIL
        image = Image.open(img_file).convert("RGB")
        img_array = np.array(image)
        
        # Perform detection
        results = model.predict(
            source=img_array,
            conf=0.25,
            imgsz=640,
            device='cpu'
        )
        
        # Get filtered results
        annotated_image, filtered_detections = draw_detections(image, results)  # Pass PIL Image directly
        st.image(annotated_image, caption="Dommages détectés (confiance ≥85%)", use_container_width=True)
        
        # Display results (unchanged)
        if filtered_detections:
            st.subheader("Dommages confirmés:")
            for det in sorted(filtered_detections, key=lambda x: x["confidence"], reverse=True):
                st.write(f"- {det['class_name']} (certitude: {det['confidence']:.0%})")
        else:
            st.warning("Aucun dommage significatif détecté")
            st.info("Conseils pour une meilleure détection:")
            st.write("• Photographiez sous un angle direct")
            st.write("• Assurez un bon éclairage")
            st.write("• Capturez les détails de près")
            
        if st.checkbox("Afficher les détails techniques"):
            st.write("Total des détections potentielles:", len(results[0].boxes))
            st.write("Détections validées (≥85%):", len(filtered_detections))
            
    except Exception as e:
        st.error(f"Erreur lors de l'analyse: {str(e)}")
