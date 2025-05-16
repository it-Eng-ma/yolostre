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
#st.write("OpenCV version:", cv2.__version__)
# Generate a random filename to send to Flutter
random_filename = f"dommages_detectes_{uuid.uuid4().hex[:8]}.png"

# French damage classes
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

#st.markdown("""
 #   <style> /* responsive CSS omitted for brevity */ </style>
#""", unsafe_allow_html=True)


st.markdown("""
    <style>
        /* Hide Streamlit UI elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* Force app to use full screen height */
        .block-container {
            padding-top: 0rem;
            padding-bottom: 0rem;
        }

        html, body, .main {
            height: 100%;
            margin: 0;
            padding: 0;
            overflow: hidden;
        }

        /* Full height for mobile too */
        .stApp {
            height: 100vh;
        }
    </style>
""", unsafe_allow_html=True)




# Load your YOLO model
MODEL_PATH = "yolostr/cardmg.pt"
try:
    model = YOLO(MODEL_PATH)
    st.success("Modèle chargé avec succès!")
except Exception as e:
    st.error(f"Erreur de chargement du modèle: {e}")
    st.stop()

#st.title("📷 Détection de Dommages sur Véhicule")
st.markdown("##### Détection de Dommages:")

def draw_detections(image, results):
    img_display = image.copy()
    detections = []
    for result in results:
        for box in result.boxes:
            conf = float(box.conf)
            #class_threshold = 0.1
           # conf_class_score = float(box.cls_conf)  # or whatever variable stores class confidence

            if conf >= 0.2:
            #if conf >= 0.9:#0.5
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


#img_file = st.file_uploader("📸 1) Prenez une photo de la partie endommagée du véhicule PUIS", type=["jpg","jpeg","png"])

       # if st.checkbox("🛠️ Afficher les détails techniques"):
            #st.write("Total des détections potentielles:", len(results[0].boxes))
           # st.write("Détections validées (≥20% confiance):", len(filtered_detections))


st.markdown("###  1) Prenez une photo📸 de la partie endommagée 🚗")
st.markdown("#### _2) Puis téléversez-la ci-dessous :_")

img_file = st.file_uploader("", type=["jpg", "jpeg", "png"])


if img_file:
    try:
        image = Image.open(img_file).convert("RGB")
       # resized_image = image.resize((920, 640))  # Keep this line
        # PIL resize uses (width, height)
        img_array = np.array(image)

        results = model.predict(
            source=img_array,
            conf=0.2,                    # Confidence threshold
            iou=0.9,                     # IoU threshold for NMS
            imgsz=(img_array.shape[0], img_array.shape[1]),  # Use original image size
            device='cpu',
            augment=True,               # Enable test-time augmentation
           # max_det=100,
            agnostic_nms=False
        )

        annotated_image, filtered_detections = draw_detections(img_array, results)
        st.image(annotated_image, caption="🛠️ Dommages détectés")  # , use_container_width=True

        # Convert annotated image to Base64 for Flutter
        buf = BytesIO()
        Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)).save(buf, format='PNG')
        b64 = base64.b64encode(buf.getvalue()).decode()

        # Send image to Flutter
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
                }} else {{
                    console.warn("⚠️ Flutter interface not found.");
                }}
            }}, 500);
            </script>
        """, height=0)

        # Send detections to Flutter
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

        if filtered_detections:
            st.subheader("✅ Dommages confirmés:")
            for det in sorted(filtered_detections, key=lambda x: x["confidence"], reverse=True):
                st.markdown(f"- **{det['class_name']}** (certitude: {det['confidence']:.0%})")
        else:
            st.warning("🚫 Aucun dommage significatif détecté")
            st.info("🔍 Conseils pour une meilleure détection :")
            st.markdown("""
                • 📸 Photographiez sous un angle direct  
                • 💡 Assurez un bon éclairage  
                • 🔍 Capturez les détails de près
            """)

    except Exception as e:
        st.error(f"❌ Erreur lors de l’analyse de l’image : {str(e)}")







