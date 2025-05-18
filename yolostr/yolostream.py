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

# Unique filename (reset only on new image)
if "random_filename" not in st.session_state:
    st.session_state.random_filename = f"dommages_detectes_{uuid.uuid4().hex[:8]}.png"

# Session state to hold results & base64 image
if "filtered_detections" not in st.session_state:
    st.session_state.filtered_detections = []
if "b64_image" not in st.session_state:
    st.session_state.b64_image = ""

# Charger modèle une seule fois
if "model" not in st.session_state:
    st.session_state.model = YOLO("yolostr/cardmg.pt")
model = st.session_state.model

# File uploader
img_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="image_upload")

if img_file is not None:
    # Nouvelle image uploadée -> réinitialiser filename pour éviter confusion
    st.session_state.random_filename = f"dommages_detectes_{uuid.uuid4().hex[:8]}.png"

    image = Image.open(img_file).convert("RGB")
    resized_image = image.resize((640, 448))
    img_array = np.array(resized_image)

    results = model.predict(
        source=img_array,
        conf=0.2,
        iou=0.7,
        device='cpu',
        imgsz=(280, 280),
        augment=True
    )

    # Dessiner détections (tu peux garder ta fonction)
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

    annotated_image, filtered_detections = draw_detections(img_array, results)

    # Sauvegarder résultats dans session_state
    st.session_state.filtered_detections = filtered_detections

    # Convertir image annotée en base64
    buf = BytesIO()
    Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)).save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode()
    st.session_state.b64_image = b64

elif img_file is None:
    # L'utilisateur a retiré l'image (clic sur X)
    st.session_state.filtered_detections = []
    st.session_state.b64_image = ""
    st.session_state.random_filename = ""

# Affichage si image détectée
if st.session_state.b64_image:
    annotated_image_bytes = base64.b64decode(st.session_state.b64_image)
    annotated_image_pil = Image.open(BytesIO(annotated_image_bytes))
    st.image(annotated_image_pil, caption="🛠️ Dommages détectés", use_column_width=True)
else:
    st.info("Aucune image sélectionnée")

# Envoi des données vers Flutter (via WebView JS handler)
# Envoi la donnée si elle existe sinon envoie une "commande" pour supprimer
payload = {
    "base64": f"data:image/png;base64,{st.session_state.b64_image}" if st.session_state.b64_image else "",
    "filename": st.session_state.random_filename if st.session_state.random_filename else "",
    "clear": True if not st.session_state.b64_image else False,
    "results": st.session_state.filtered_detections
}
payload_json = json.dumps(payload)

components.html(f"""
    <script>
    setTimeout(function() {{
        if (window.flutter_inappwebview) {{
            window.flutter_inappwebview.callHandler('sendAnnotatedImage', {json.dumps(payload)});
        }}
    }}, 500);
    </script>
""", height=0)

# Affichage texte résultats
if st.session_state.filtered_detections:
    st.subheader("✅ Dommages détectés :")
    for det in sorted(st.session_state.filtered_detections, key=lambda x: x["confidence"], reverse=True):
        st.markdown(f"- **{det['class_name']}** (certitude: {det['confidence']:.0%})")
else:
    st.warning("🚫 Aucun dommage significatif détecté")
