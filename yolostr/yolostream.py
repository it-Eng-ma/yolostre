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

# Générer un nom de fichier unique
random_filename = f"dommages_detectes_{uuid.uuid4().hex[:8]}.png"

# Dictionnaire des classes
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

# Cacher les éléments UI Streamlit
st.markdown("""
    <style>
        #MainMenu, footer, header {visibility: hidden;}
        .block-container {padding-top: 0rem; padding-bottom: 0rem;}
        html, body, .main, .stApp {height: 100vh; margin: 0; padding: 0; overflow: hidden;}
    </style>
""", unsafe_allow_html=True)

# Charger le modèle
MODEL_PATH = "yolostr/cardmg.pt"
try:
    model = YOLO(MODEL_PATH)
    st.success("✅ Modèle YOLO chargé avec succès.")
except Exception as e:
    st.error(f"❌ Erreur de chargement du modèle: {e}")
    st.stop()

# Instructions utilisateur
st.markdown("### 1) Prenez une photo 📸 de la zone endommagée")
st.markdown("#### 2) Puis téléversez-la ci-dessous :")
img_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

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

if img_file:
    try:
        image = Image.open(img_file).convert("RGB")

        # Redimensionner à la taille d'entrée du modèle (640x448)
        resized_image = image.resize((640, 448))
        img_array = cv2.cvtColor(np.array(resized_image), cv2.COLOR_RGB2BGR)

        # Prédiction
        results = model.predict(
            source=img_array,
            conf=0.2,
            iou=0.3,
            device='cpu',
            imgsz=(448, 640),
            augment=False
        )

        # Dessiner les résultats
        annotated_image, filtered_detections = draw_detections(img_array, results)
        st.image(annotated_image, caption="🛠️ Dommages détectés")

        # Convertir l'image annotée en base64
        buf = BytesIO()
        Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)).save(buf, format='PNG')
        b64 = base64.b64encode(buf.getvalue()).decode()

        # Envoyer l'image annotée vers Flutter
        components.html(f"""
            <script>
            setTimeout(function() {{
                const payload = {{
                    base64: "data:image/png;base64,{b64}",
                    filename: "{random_filename}"
                }};
                if (window.flutter_inappwebview) {{
                    window.flutter_inappwebview.callHandler('sendAnnotatedImage', payload)
                        .then(res => console.log("✅ Annotated image sent", res));
                }}
            }}, 500);
            </script>
        """, height=0)

        # Envoyer les résultats en JSON vers Flutter
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

        # Résultats lisibles par l'utilisateur
        if filtered_detections:
            st.subheader("✅ Dommages détectés :")
            for det in sorted(filtered_detections, key=lambda x: x["confidence"], reverse=True):
                st.markdown(f"- **{det['class_name']}** (certitude : {det['confidence']:.0%})")
        else:
            st.warning("🚫 Aucun dommage significatif détecté.")
            st.info("🔍 Astuces pour une meilleure détection :")
            st.markdown("""
                • 📸 Photographiez directement la zone  
                • 💡 Assurez un bon éclairage  
                • 🔍 Capturez les détails de près
            """)

    except Exception as e:
        st.error(f"❌ Erreur lors de l’analyse de l’image : {str(e)}")
