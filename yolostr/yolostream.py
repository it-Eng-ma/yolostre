import streamlit as st
from PIL import Image, ExifTags
import numpy as np
import cv2
from ultralytics import YOLO
import json
import streamlit.components.v1 as components
import uuid
import base64
from io import BytesIO

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

# 1) Cache the YOLO model so it's loaded only once
@st.cache_resource
def load_model():
    return YOLO("yolostr/cardmg.pt")

model = load_model()

# 2) EXIF orientation fixer
def apply_exif_orientation(img: Image.Image) -> Image.Image:
    try:
        exif = img._getexif()
        if exif:
            for tag, val in exif.items():
                if ExifTags.TAGS.get(tag) == "Orientation":
                    if val == 3:
                        img = img.rotate(180, expand=True)
                    elif val == 6:
                        img = img.rotate(270, expand=True)
                    elif val == 8:
                        img = img.rotate(90, expand=True)
                    break
    except Exception:
        pass
    return img

# 3) Draw detection boxes
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

def draw_detections(image: np.ndarray, results) -> tuple[np.ndarray, list]:
    img_disp = image.copy()
    dets = []
    for res in results:
        for box in res.boxes:
            conf = float(box.conf)
            if conf < 0.2:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls)
            name = CLASS_NAMES.get(cls_id, f"inconnu {cls_id}")
            cv2.rectangle(img_disp, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(img_disp,
                        f"{name} {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0,0,255),
                        2)
            dets.append({
                "class_name": name,
                "confidence": round(conf, 2),
                "coords": [x1, y1, x2, y2]
            })
    return img_disp, dets

# 4) Session-state initialization
if "last_file_id" not in st.session_state:
    st.session_state.last_file_id = None
if "detections" not in st.session_state:
    st.session_state.detections = []
if "b64_image" not in st.session_state:
    st.session_state.b64_image = ""
if "random_filename" not in st.session_state:
    st.session_state.random_filename = ""

# 5) UI styling
st.markdown("""
    <style>
        #MainMenu, footer, header {visibility: hidden;}
        .block-container {padding-top: 0; padding-bottom: 0;}
        html, body, .main, .stApp {height:100vh; margin:0; padding:0; overflow:hidden;}
    </style>
""", unsafe_allow_html=True)

st.markdown("### 1) Prenez ou sélectionnez une photo de la partie endommagée")
st.markdown("#### 2) Puis téléversez-la ci-dessous :")

# 6) File uploader
img_file = st.file_uploader("", type=["jpg","jpeg","png"], key="image_upload")
file_id = img_file.name if img_file else None

# 7) If the user changed or cleared the upload, reset state and rerun
if file_id != st.session_state.last_file_id:
    st.session_state.last_file_id = file_id
    st.session_state.detections = []
    st.session_state.b64_image = ""
    st.session_state.random_filename = (f"dommages_detectes_{uuid.uuid4().hex[:8]}.png"
                                        if img_file else "")
    st.experimental_rerun()

# 8) When a new file is present and no detections yet, run inference
if img_file and not st.session_state.detections:
    # Load + fix orientation + convert + resize
    pil_img = Image.open(img_file)
    pil_img = apply_exif_orientation(pil_img)
    pil_img = pil_img.convert("RGB")
    pil_img = pil_img.resize((448, 640), Image.ANTIALIAS)

    img_arr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # Inference (no augment for speed)
    results = model.predict(
        source=img_arr,
        conf=0.2,
        iou=0.3,
        device='cpu',
        imgsz=(448, 640),
        augment=False
    )

    annotated, dets = draw_detections(img_arr, results)
    st.session_state.detections = dets

    # Encode annotated image to base64
    buf = BytesIO()
    Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)).save(buf, format="PNG")
    st.session_state.b64_image = base64.b64encode(buf.getvalue()).decode()
    st.session_state.random_filename = f"dommages_detectes_{uuid.uuid4().hex[:8]}.png"

# 9) Display the annotated image or placeholder
if st.session_state.b64_image:
    img_to_show = Image.open(BytesIO(base64.b64decode(st.session_state.b64_image)))
    st.image(img_to_show, caption="🛠️ Dommages détectés", use_column_width=True)
else:
    st.info("Aucune image sélectionnée")

# 10) Send results to Flutter WebView
payload = {
    "base64": f"data:image/png;base64,{st.session_state.b64_image}",
    "filename": st.session_state.random_filename,
    "results": st.session_state.detections,
    "clear": not bool(st.session_state.b64_image)
}
components.html(f"""
<script>
  setTimeout(() => {{
    window.flutter_inappwebview?.callHandler('sendResults', {json.dumps(payload)});
  }}, 100);
</script>
""", height=0)

# 11) Text summary of detections
if st.session_state.detections:
    st.subheader("✅ Dommages détectés :")
    for d in sorted(st.session_state.detections, key=lambda x: x["confidence"], reverse=True):
        st.markdown(f"- **{d['class_name']}** ({d['confidence']:.0%})")
else:
    st.warning("🚫 Aucun dommage significatif détecté")
    st.info("🔍 Conseils : prenez la photo bien centrée, bon éclairage et de pr











