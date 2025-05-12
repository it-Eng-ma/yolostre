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

# Generate a random filename for our annotated image
random_filename = f"dommages_detectes_{uuid.uuid4().hex[:8]}.png"

# French labels
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

st.markdown("""
    <style>/* mobile‐friendly CSS */</style>
""", unsafe_allow_html=True)

# Load your YOLO model
model = YOLO("yolostr/cardmg.pt")

st.title("📷 Détection de Dommages sur Véhicule")

def draw_detections(image, results):
    img_display = image.copy()
    detections = []
    for result in results:
        for box in result.boxes:
            conf = float(box.conf)
            if conf < 0.5: continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls)
            name = CLASS_NAMES.get(cls_id, f"inconnu {cls_id}")
            cv2.rectangle(img_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_display, f"{name} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            detections.append({
                "class_name": name,
                "confidence": conf,
                "coords": [x1, y1, x2, y2]
            })
    return img_display, detections

img_file = st.file_uploader("📁 Téléversez une image", type=["jpg","jpeg","png"])
if img_file:
    image = Image.open(img_file).convert("RGB")
    img_arr = np.array(image)
    results = model.predict(source=img_arr, conf=0.5, imgsz=640, device='cpu')
    annotated, dets = draw_detections(img_arr, results)
    st.image(annotated, caption="🛠️ Dommages détectés", use_container_width=True)

    # Encode annotated image to base64
    pil_annot = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    buf = BytesIO()
    pil_annot.save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode()

    # Send both the base64 data AND the filename to Flutter
    components.html(f"""
      <script>
        setTimeout(() => {{
          const data = 'data:image/png;base64,{b64}';
          const filename = '{random_filename}';
          if (window.flutter_inappwebview) {{
            window.flutter_inappwebview
              .callHandler('sendAnnotatedImage', data, filename)
              .then(res => console.log('✅ sent image + name:', res));
          }} else console.warn('⚠️ Flutter handler missing');
        }}, 500);
      </script>
    """, height=0)

    # Send the detection results too
    if dets:
      st.subheader("✅ Dommages confirmés:")
      for d in sorted(dets, key=lambda x: x["confidence"], reverse=True):
          st.markdown(f"- **{d['class_name']}** ({d['confidence']:.0%})")
      comp_js = json.dumps(dets)
      components.html(f"""
        <script>
          setTimeout(() => {{
            const results = {comp_js};
            window.flutter_inappwebview
              .callHandler('sendResults', results)
              .then(res => console.log('✅ sent results:', res));
          }}, 500);
        </script>
      """, height=0)
    else:
      st.warning("🚫 Aucun dommage significatif détecté")
