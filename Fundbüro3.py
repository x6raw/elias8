import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Titel der App
st.title("🔍 KI-Fundbüro")
st.write("Lade ein Bild eines gefundenen Gegenstands hoch.")

# Modell laden
@st.cache_resource
def load_my_model():
    return load_model("keras_Model.h5", compile=False)

model = load_my_model()

# Labels laden
@st.cache_data
def load_labels():
    return open("labels.txt", "r").readlines()

class_names = load_labels()

# Bild-Upload
uploaded_file = st.file_uploader("📷 Bild hochladen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Bild anzeigen
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", use_container_width=True)

    # Bild vorbereiten
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # Array erstellen
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Bild in Array umwandeln
    image_array = np.asarray(image)

    # Normalisieren
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data[0] = normalized_image_array

    # Vorhersage
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    # Ergebnis anzeigen
    st.subheader("📊 Ergebnis")

    st.write(f"**Erkanntes Objekt:** {class_name}")
    st.write(f"**Sicherheit:** {round(confidence_score * 100, 2)} %")

    # Fundbüro-Logik
    st.subheader("📦 Fundbüro-Eintrag")

    if confidence_score > 0.7:
        st.success(f"Der Gegenstand wurde als **{class_name}** erkannt und gespeichert.")
    else:
        st.warning("Unsichere Erkennung – bitte manuell überprüfen.")

    # Optional: einfache Datenbank (Session)
    if "items" not in st.session_state:
        st.session_state.items = []

    if st.button("➕ Ins Fundbüro aufnehmen"):
        st.session_state.items.append({
            "name": class_name,
            "confidence": float(confidence_score)
        })
        st.success("Eintrag gespeichert!")

# Fundliste anzeigen
st.subheader("📋 Aktuelles Fundbüro")

if "items" in st.session_state and len(st.session_state.items) > 0:
    for i, item in enumerate(st.session_state.items):
        st.write(f"{i+1}. {item['name']} ({round(item['confidence']*100, 2)}%)")
else:
    st.info("Noch keine Einträge vorhanden.")
