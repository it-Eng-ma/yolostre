import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import json
import streamlit.components.v1 as components
import uuid
import base64
from io import BytesIO

# Nom de fichier unique
random_filename = f"dommages_detectes_{uuid.uuid4().hex[:8]}.png"

# Labels des classes
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

# Cacher UI Streamlit
st.markdown("""
    <style>
        #MainMenu, footer, header {visibility: hidden;}
        .block-container {padding-top: 0rem; padding-bottom: 0rem;}
        html, body, .main, .stApp {height: 100vh; margin: 0; padding: 0; overflow: hidden;}
    </style>
""", unsafe_allow_html=True)

# Charger le modèle YOLO une seule fois (session state)
if "model" not in st.session_state:
    try:
        st.session_state.model = YOLO("yolostr/cardmg.pt")
        st.success("✅ Modèle YOLO chargé avec succès")
    except Exception as e:
        st.error(f"❌ Erreur de chargement du modèle: {e}")
        st.stop()

model = st.session_state.model

# Instructions
st.markdown("### 1) Prenez une photo 📸 de la partie endommagée 🚗")
st.markdown("#### _2) Puis téléversez-la ci-dessous :_")

# Uploader image
img_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="image_upload")

def draw_detections(image, results):
    img_display = image.copy()
    detections = []
    for result in results:
        for box in result.boxes:
            conf = float(box.conf)
            if conf >= 0.2:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls)
                name = CLASS_NAMES.get(cls_id, f"inconnu {cls_id}")
                cv2.rectangle(img_display, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(img_display, f"{name} {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                detections.append({
                    "class_name": name,
                    "confidence": conf,
                    "coords": [x1, y1, x2, y2]
                })
    return img_display, detections

# Traitement après upload
if img_file is not None:
    try:
        image = Image.open(img_file).convert("RGB")

        resized_image = image.resize((650 ,400))
        img_array = np.array(resized_image)

        results = model.predict(
            source=img_array,
            conf=0.2,
            iou=0.3,
            device='cpu',
            imgsz=(225, 225),
            #augment=True
        )

        annotated_image, filtered_detections = draw_detections(img_array, results)

        st.image(annotated_image, caption="🛠️ Dommages détectés", use_column_width=True)

        # Encoder image pour Flutter (base64)
        buf = BytesIO()
        Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)).save(buf, format='PNG')
        b64 = base64.b64encode(buf.getvalue()).decode()

        # JS pour Flutter WebView - Image
        components.html(f"""
            <script>
            setTimeout(function() {{
                const payload = {{
                    base64: "data:image/png;base64,{b64}",
                    filename: "{random_filename}"
                }};
                if (window.flutter_inappwebview) {{
                    window.flutter_inappwebview.callHandler('sendAnnotatedImage', payload)
                        .then(res => console.log("✅ Image envoyée", res));
                }}
            }}, 500);
            </script>
        """, height=0)

        # JS pour Flutter WebView - Résultats
        results_json = json.dumps(filtered_detections)
        components.html(f"""
            <script>
            setTimeout(function() {{
                if (window.flutter_inappwebview) {{
                    window.flutter_inappwebview.callHandler('sendResults', {results_json});
                }}
            }}, 500);
            </script>
        """, height=0)

        # Résumé des résultats
        if filtered_detections:
            st.subheader("✅ Dommages détectés :")
            for det in sorted(filtered_detections, key=lambda x: x["confidence"], reverse=True):
                st.markdown(f"- **{det['class_name']}** (certitude: {det['confidence']:.0%})")
        else:
            st.warning("🚫 Aucun dommage significatif détecté")
            st.info("🔍 Astuces pour de meilleurs résultats :")
            st.markdown("""
                • 📸 Prendre la photo bien centrée  
                • 💡 Assurez un bon éclairage  
                • 🔍 Photographiez de près
            """)

    except Exception as e:
        st.error(f"❌ Erreur lors du traitement de l’image : {str(e)}")
