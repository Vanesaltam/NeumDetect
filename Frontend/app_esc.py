import streamlit as st
import requests
from PIL import Image
import io, os

# Configuración de la página
st.set_page_config(page_title="NeumDetect", layout="wide")

# Funciones de predicción para cada modelo
def predict_vgg16(image):
    return send_prediction_request(image, "vgg16")

def predict_resnet50(image):
    return send_prediction_request(image, "resnet50")

def predict_custom(image):
    return send_prediction_request(image, "custom")

def send_prediction_request(image, model_name):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    files = {'file': ('image.png', img_byte_arr, 'image/png')}
    response = requests.post(f"http://localhost:8000/predict_{model_name}", files=files)

    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error al procesar la imagen con el modelo {model_name}. Por favor, intente de nuevo.")
        return None

# Diccionario de modelos disponibles
MODELS = {
    "Modelo VGG16": predict_vgg16,
    "Modelo ResNet50": predict_resnet50,
    "Modelo personalizado": predict_custom
}

# Función para mostrar la imagen centrada
def display_centered_image(image_path, width=300):
    col1, col2, col3 = st.columns([1,3,1])
    with col2:
        st.image(image_path, width=width,use_column_width=False)

current_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(current_dir, '..', 'img', 'LOGO2.png')

# Mostrar la imagen centrada en el encabezado
display_centered_image(logo_path, width=300)  # Ajusta la ruta y el ancho según sea necesario

# Título y logo
st.title("NeumDetect: Detección de Neumonía")
st.write("Plataforma de detección de neumonía basada en imágenes de rayos X de tórax")

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.subheader("Acerca de NeumDetect")
    st.write("NeumDetect es una aplicación web que utiliza modelos de clasificación de redes neuronales para detectar la presencia de neumonía en imágenes de rayos X de tórax.")
    st.write("Los modelos disponibles son VGG16, ResNet50, (...).")
    st.write("La aplicación fue desarrollada como parte de un proyecto de aprendizaje automático.")
    # st.write("Para obtener más información, visita el repositorio de GitHub.")
    # st.write("[Repositorio de GitHub](https://github.com/PabloZuVal/NeumDetect)")

with col2:
    st.subheader("Instrucciones")
    st.write("1. Carga una imagen de rayos X de tórax en formato .jpg o .png.")
    st.write("2. Selecciona un modelo de clasificación.")
    st.write("3. Haz clic en el botón 'Realizar predicción' para obtener el diagnóstico.")

    # Carga de imagen
    uploaded_file = st.file_uploader("Cargar imagen de rayos X", type=["jpg", "jpeg", "png"])
with col3:
    st.subheader("Modelos de clasificación")
    st.write("Selecciona un modelo de clasificación para realizar la predicción.")
    st.write("Cada modelo utiliza una arquitectura de red neuronal diferente para clasificar las imágenes.")
    # Selección de modelo
    model_choice = st.selectbox("Seleccionar modelo", list(MODELS.keys()))

    # Botón para realizar la predicción
    if st.button("Realizar predicción"):
        if uploaded_file is not None:
            image = Image.open(uploaded_file)

            # Realizar predicción con el modelo seleccionado
            result = MODELS[model_choice](image)

            if result:
                # col1, col2 = st.columns([1, 2])

                # with col1:
                st.image(image, caption="Imagen cargada", width=300)

                st.write(f"Diagnóstico: {result['diagnosis']}")
                st.write(f"Probabilidad: {result['probability']}")
                # with col2:
        else:
            st.write("Por favor, carga una imagen antes de realizar la predicción.")
