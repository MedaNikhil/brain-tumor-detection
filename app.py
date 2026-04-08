import tensorflow as tf
import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import gdown
import os

# -------------------------------
# PAGE SETTINGS
# -------------------------------
st.set_page_config(page_title="Brain Tumor Detection", layout="wide")

st.markdown(
    "<h1 style='text-align: center; color: cyan;'>🧠 Brain Tumor Detection System</h1>",
    unsafe_allow_html=True
)

# -------------------------------
# DOWNLOAD MODEL (SAFE)
# -------------------------------
MODEL_URL = "https://drive.google.com/uc?id=14ooqw2ANOG2zikNLKOSizRNhjA_C9zwJ"

@st.cache_resource
def download_and_load_model():
    if not os.path.exists("model.h5"):
        with st.spinner("⬇️ Downloading model..."):
            gdown.download(MODEL_URL, "model.h5", quiet=False)
    return tf.keras.models.load_model("model.h5")

model = download_and_load_model()

# -------------------------------
# CLASS NAMES
# -------------------------------
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# -------------------------------
# INPUT SECTION
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("📤 Upload MRI Image", type=["jpg", "png", "jpeg"])

with col2:
    predict = st.button("🔍 Predict")

# -------------------------------
# SHOW IMAGE
# -------------------------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="MRI Image", width=250)

# -------------------------------
# PREDICTION
# -------------------------------
if predict and uploaded_file:

    # PREPROCESS
    img = image.resize((128,128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # PREDICT
    with st.spinner("🔍 Analyzing MRI image..."):
        prediction = model.predict(img)

    # RESULT
    class_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    tumor_type = class_names[class_index]

    # LOW CONFIDENCE WARNING
    if confidence < 70:
        st.warning("⚠️ Low confidence prediction. Try another image.")

    # DISPLAY RESULT
    st.success(f"🧠 Tumor Type: {tumor_type}")
    st.info(f"📊 Confidence: {confidence:.2f}%")

    # -------------------------------
    # 🔍 PREDICTION BREAKDOWN (IMPROVED UI)
    # -------------------------------
    st.markdown("### 🔍 Prediction Breakdown")

    for i, val in enumerate(prediction[0]):
        st.progress(float(val))  # progress bar
        st.write(f"{class_names[i]}: {val*100:.2f}%")

    # -------------------------------
    # GRAPH DATA (STATIC)
    # -------------------------------
    accSVM = 89.36
    accKNN = 82.98
    accCNN = 91.49

    precision = 0.92
    recall = 0.91
    specificity = 0.97
    f1 = 0.92

    col3, col4 = st.columns(2)

    # GRAPH
    with col3:
        fig, ax = plt.subplots()

        models = ['SVM', 'KNN', 'CNN']
        values = [accSVM, accKNN, accCNN]

        ax.bar(models, values)

        best_index = np.argmax(values)
        ax.text(best_index, values[best_index] + 2, "🏆 Best", ha='center')

        ax.set_ylim(0, 100)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("📊 Model Comparison")

        st.pyplot(fig)

    # METRICS
    with col4:
        st.markdown(f"""
        ### 📈 Metrics

        - SVM Accuracy: {accSVM}%
        - KNN Accuracy: {accKNN}%
        - CNN Accuracy: {accCNN}%

        ---
        - Precision: {precision}
        - Recall: {recall}
        - Specificity: {specificity}
        - F1 Score: {f1}
        """)

# -------------------------------
# PROJECT NOTE
# -------------------------------
st.markdown("""
---
### 📌 Project Note

- CNN model is used for brain tumor classification.
- Prediction probabilities vary for each input image.
- Accuracy metrics are calculated during training.
- Future improvements include better dataset and model tuning.
""")
