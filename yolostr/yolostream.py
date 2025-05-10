import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO

import json
import streamlit.components.v1 as components
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

# Custom CSS for better mobile experience
st.markdown("""
    <style>
        /* Better padding and font scaling for small screens */
        html, body, [class*="css"]  {
            font-size: 16px;
        }

        /* Ensure uploader and image are fully responsive */
        .element-container {
            width: 100% !important;
        }

        img {
            max-width: 100%;
            height: auto;
        }

        /* Mobile-friendly buttons and checkboxes */
        button, .stButton>button, .stCheckbox>label {
            font-size: 1.1em;
        }
    </style>
""", unsafe_allow_html=True)

# Load model
MODEL_PATH = "yolostr/cardmg.pt"
try:
    model = YOLO(MODEL_PATH)
    st.success("Modèle chargé avec succès!")
except Exception as e:
    st.error(f"Erreur de chargement du modèle: {str(e)}")
    st.stop()

st.title("📷 Détection de Dommages sur Véhicule")

def draw_detections(image, results):
    img_display = image.copy()
    detections = []
    
    for result in results:
        for box in result.boxes:
            conf = float(box.conf)
            if conf >= 0.85:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls)
                class_name = CLASS_NAMES.get(cls_id, f"inconnu {cls_id}")
                
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

img_file = st.file_uploader("📁 Téléversez une image du véhicule", type=["jpg", "jpeg", "png"])

if img_file:
    try:
        image = Image.open(img_file).convert("RGB")
        img_array = np.array(image)
        
        results = model.predict(
            source=img_array,
            conf=0.25,
            imgsz=640,
            device='cpu'
        )
        
        annotated_image, filtered_detections = draw_detections(img_array, results)
        st.image(annotated_image, caption="🛠️ Dommages détectés (confiance ≥85%)", use_container_width=True)
        
        if filtered_detections:
            st.subheader("✅ Dommages confirmés:")
            for det in sorted(filtered_detections, key=lambda x: x["confidence"], reverse=True):
                st.markdown(f"- **{det['class_name']}** (certitude: {det['confidence']:.0%})")

            results_json = json.dumps(filtered_detections)

            components.html(f"""
                <script>
                setTimeout(function() {{
                    const results = {results_json};

                    if (window.flutter_inappwebview) {{
                        window.flutter_inappwebview.callHandler('sendResults', results)
                            .then(function(response) {{
                                console.log("✅ Results sent to Flutter:", response);
                            }});
                    }} else {{
                        console.warn("⚠️ Flutter interface not found.");
                    }}
                }}, 1000); // Delay to ensure Flutter is ready
                </script>
            """, height=0)

        else:
            st.warning("🚫 Aucun dommage significatif détecté")
            st.info("🔍 Conseils pour une meilleure détection :")
            st.markdown("""
                • 📸 Photographiez sous un angle direct  
                • 💡 Assurez un bon éclairage  
                • 🔍 Capturez les détails de près
            """)
        
        if st.checkbox("🛠️ Afficher les détails techniques"):
            st.write("Total des détections potentielles:", len(results[0].boxes))
            st.write("Détections validées (≥85%):", len(filtered_detections))
            
    except Exception as e:
        st.error(f"Erreur lors de l'analyse: {str(e)}")




# Convert filtered detections to JSON string (only if detections exist)

  # Height = 0 to hide it from UI




















