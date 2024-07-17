import io
import os
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers, models
from PIL import Image
import logging
from google.cloud import aiplatform
from google.cloud.aiplatform.gapic.schema import predict
from google.protobuf import json_format
from dotenv import load_dotenv
import base64
import tempfile

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

app = FastAPI()

# Configuración de Vertex AI
PROJECT_ID = os.getenv('PROJECT_ID')
ENDPOINT_ID = os.getenv('ENDPOINT_ID')
LOCATION = os.getenv('LOCATION')

# Inicializar el cliente de Vertex AI
aiplatform.init(project=PROJECT_ID, location=LOCATION)

# Configuración del modelo VGG16
current_dir = os.path.dirname(os.path.abspath(__file__))
weights_path = os.path.join(current_dir, '..', 'Models', 'VGG16', 'modelo_neumonia_vgg16_precision_weights.h5')

def create_model():
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(150, 150, 3))
    base_model.trainable = False

    flatten_layer = layers.Flatten()
    dense_layer1 = layers.Dense(500, activation='relu')
    dense_layer2 = layers.Dense(250, activation='relu')  # Nueva capa densa
    dense_layer3 = layers.Dense(100, activation='relu')  # Nueva capa densa
    prediction_layer = layers.Dense(1, activation='sigmoid')

    model = models.Sequential([
        base_model,
        flatten_layer,
        dense_layer1,
        dense_layer2,
        dense_layer3,
        prediction_layer
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = create_model()
try:
    model.load_weights(weights_path)
    logger.info(f"Modelo VGG16 cargado exitosamente desde {weights_path}")
except Exception as e:
    logger.error(f"Error al cargar el modelo VGG16: {str(e)}")
    raise

def load_and_preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert('RGB')  # Convertir a RGB si es escala de grises
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.post("/predict_vgg16")
async def predict_vgg16(file: UploadFile = File(...)):
    try:
        logger.info(f"Recibida imagen: {file.filename}")
        image_data = await file.read()
        img = Image.open(io.BytesIO(image_data))

        logger.info(f"Imagen cargada. Modo: {img.mode}, Tamaño: {img.size}")
        img_array = load_and_preprocess_image(img)

        logger.info(f"Imagen preprocesada. Shape: {img_array.shape}")
        prediction = model.predict(img_array)

        logger.info(f"Predicción realizada. Valor: {prediction[0][0]}")

        if prediction[0][0] > 0.5:
            result = "Neumonía"
            probability = (prediction[0][0])
        else:
            result = "Normal"
            probability = (1 - prediction[0][0])

        return {
            "diagnosis": result,
            "probability": f"{probability:.2f}"
        }
    except Exception as e:
        logger.error(f"Error durante la predicción: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def predict_vertex_ai(filename: str):
    client_options = {"api_endpoint": f"{LOCATION}-aiplatform.googleapis.com"}
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)

    with open(filename, "rb") as f:
        file_content = f.read()

    encoded_content = base64.b64encode(file_content).decode("utf-8")
    instance = predict.instance.ImageClassificationPredictionInstance(
        content=encoded_content,
    ).to_value()
    instances = [instance]

    endpoint = client.endpoint_path(project=PROJECT_ID, location=LOCATION, endpoint=ENDPOINT_ID)
    response = client.predict(endpoint=endpoint, instances=instances)

    return response

@app.post("/predict_vertex")
async def predict_vertex(file: UploadFile = File(...)):
    try:
        logger.info(f"Iniciando predicción Vertex AI para archivo: {file.filename}")

        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(contents)
            temp_filename = temp_file.name

        prediction = predict_vertex_ai(temp_filename)
        result = json_format.MessageToDict(prediction._pb)
        predictions = result['predictions'][0]['confidences']

        logger.info(f"Resultado de la predicción Vertex AI: {result}")

        diagnoses = ["Normal", "Neumonía"]
        max_confidence_index = predictions.index(max(predictions))
        diagnosis = diagnoses[max_confidence_index]
        probability = predictions[max_confidence_index]

        return {
            "diagnosis": diagnosis,
            "probability": f"{probability:.2f}"
        }

    except Exception as e:
        logger.error(f"Error durante la predicción Vertex AI: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
