import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Función para cargar y preprocesar la imagen
def load_and_prep_image(image, img_shape=224):
    img = tf.image.decode_image(image, channels=3)
    img = tf.image.resize(img, [img_shape, img_shape])
    img = img/255.
    return img

# Función para realizar la predicción
def predict_class(model, image):
    img_array = load_and_prep_image(image)
    img_array = tf.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return "Neumonía" if prediction[0][0] > 0.5 else "Normal"

# Configuración de la página
st.set_page_config(page_title="NeumDetect", layout="wide")

# Título y logo
st.title("NeumDetect: Detección de Neumonía")
st.write("Plataforma de detección de neumonía basada en imágenes de rayos X de tórax")

# Carga de imagen
uploaded_file = st.file_uploader("Cargar imagen de rayos X", type=["jpg", "jpeg", "png"])

# Selección de modelo
model_choice = st.selectbox(
    "Seleccionar modelo",
    ("Modelo VGG16", "Modelo ResNet50", "Modelo personalizado")
)

# Botón para realizar la predicción
if st.button("Realizar predicción"):
    if uploaded_file is not None:
        # Aquí iría la lógica para cargar el modelo seleccionado
        # Por ahora, usaremos un modelo ficticio
        model = tf.keras.Sequential([tf.keras.layers.Dense(1, activation="sigmoid")])

        # Realizar la predicción
        image = uploaded_file.read()
        prediction = predict_class(model, image)

        # Mostrar la imagen y el resultado
        st.image(Image.open(uploaded_file), caption="Imagen cargada", use_column_width=True)
        st.write(f"Predicción: {prediction}")
    else:
        st.write("Por favor, carga una imagen antes de realizar la predicción.")
