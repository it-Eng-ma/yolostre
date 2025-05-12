# ── app.py ──
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

# Generate a random filename
random_filename = f"dommages_detectes_{uuid.uuid4().hex[:8]}.png"

# Damage class names
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

# Load the YOLO model
MODEL_PATH = "yolostr/cardmg.pt"
try:
    model = YOLO(MODEL_PATH)
    st.success("Modèle chargé avec succès!")
except Exception as e:
    st.error(f"Erreur de chargement du modèle: {e}")
    st.stop()

# UI
st.markdown("### 1) Prenez une photo📸 de la partie endommagée 🚗")
st.markdown("#### _2) Puis téléversez-la ci-dessous :_")

# Inject custom file input to avoid mobile camera pop-up
components.html("""
  <input type="file" id="fileUpload" accept="image/png,image/jpeg" style="font-size: 18px; margin-top: 10px;">
  <script>
    const upload = document.getElementById("fileUpload");
    upload.addEventListener("change", function() {
      const file = this.files[0];
      const reader = new FileReader();
      reader.onload = function() {
        const base64Data = reader.result.split(",")[1];
        window.parent.postMessage({
          type: "uploadedImage",
          base64: base64Data,
          filename: file.name
        }, "*");
      };
      reader.readAsDataURL(file);
    });
  </script>
""", height=50)

# Use Streamlit-JavaScript to capture the uploaded image (pip install streamlit-javascript)
from streamlit_javascript import st_javascript

uploaded = st_javascript("""
window._uploadedImageData = null;
window.addEventListener("message", (e) => {
  if (e.data?.type === "uploadedImage") {
    window._uploadedImageData = e.data;
  }
});
if (window._uploadedImageData) {
  JSON.stringify(window._uploadedImageData);
} else {
  null;
}
""")

# Only continue if user uploaded an image
if uploaded:
    uploaded = json.loads(uploaded)
    img_bytes = base64.b64decode(uploaded["base64"])
    image = Image.open(BytesIO(img_bytes)).convert("RGB")

    arr = np.array(image)
    results = model.predict(source=arr, conf=0.5, imgsz=640, device='cpu')
    
    # Detection
    def draw_detections(image, results):
        img_display = image.copy()
        detections = []
        for result in results:
            for box in result.boxes:
                conf = float(box.conf)
                if conf >= 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cls_id = int(box.cls)
                    name = CLASS_NAMES.get(cls_id, f"inconnu {cls_id}")
                    cv2.rectangle(img_display, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(img_display, f"{name} {conf:.2f}",
                                (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                    detections.append({
                        "class_name": name,
                        "confidence": conf,
                        "coords": [x1, y1, x2, y2]
                    })
        return img_display, detections

    annotated, dets = draw_detections(arr, results)
    st.image(annotated, caption="🛠️ Dommages détectés", use_container_width=True)

    # Convert annotated image to base64
    buf = BytesIO()
    Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)).save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode()

    # Send image to Flutter
    components.html(f"""
      <script>
        setTimeout(function(){{
          const payload = {{
            base64: "data:image/png;base64,{b64}",
            filename: "{random_filename}"
          }};
          if (window.flutter_inappwebview) {{
            window.flutter_inappwebview
              .callHandler('sendAnnotatedImage', payload)
              .then(r => console.log("✅ Annotated sent", r));
          }} else {{
            console.warn("⚠️ Flutter interface missing");
          }}
        }}, 500);
      </script>
    """, height=0)

    # Send detections to Flutter
    if dets:
        js = json.dumps(dets)
        components.html(f"""
          <script>
            setTimeout(function(){{
              window.flutter_inappwebview
                .callHandler('sendResults', {js});
            }}, 500);
          </script>
        """, height=0)
    else:
        st.warning("🚫 Aucun dommage significatif détecté")
