from fastapi import FastAPI, File, UploadFile, HTTPException
from google.cloud import aiplatform
from google.cloud.aiplatform.gapic.schema import predict
from google.protobuf import json_format
import base64
import io
import tempfile
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configuración del proyecto y endpoint
PROJECT_ID = "neumdetect-428422"
ENDPOINT_ID = "7249016390252756992"
LOCATION = "us-central1"

# Inicializar el cliente de Vertex AI
aiplatform.init(project=PROJECT_ID, location=LOCATION)

# Create a logger instance
# logger = logging.getLogger(__name__)

def predict_image_classification_sample(
    project: str,
    endpoint_id: str,
    location: str,
    filename: str
):
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": f"{location}-aiplatform.googleapis.com"}
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)

    with open(filename, "rb") as f:
        file_content = f.read()

    # The format of each instance should conform to the deployed model's prediction input schema.
    encoded_content = base64.b64encode(file_content).decode("utf-8")
    instance = predict.instance.ImageClassificationPredictionInstance(
        content=encoded_content,
    ).to_value()
    instances = [instance]

    endpoint = client.endpoint_path(project=project, location=location, endpoint=endpoint_id)
    response = client.predict(endpoint=endpoint, instances=instances)

    return response

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):

    logger.info(f"Iniciando predicción para archivo: {file.filename}")

    try:

        # Leer el contenido del archivo
        contents = await file.read()

        # Guardar temporalmente el archivo
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(contents)
            temp_filename = temp_file.name

        # Realizar la predicción
        prediction = predict_image_classification_sample(
            project=PROJECT_ID,
            endpoint_id=ENDPOINT_ID,
            location=LOCATION,
            filename=temp_filename
        )

        # Procesar el resultado
        result = json_format.MessageToDict(prediction._pb)
        predictions = result['predictions'][0]['confidences']

        logger.info(f"Resultado de la predicción: {result}")

        # Interpretar los resultados
        diagnoses = ["Normal", "Neumonía"]
        max_confidence_index = predictions.index(max(predictions))
        diagnosis = diagnoses[max_confidence_index]
        probability = predictions[max_confidence_index]

        return {
            "diagnosis": diagnosis,
            "probability": f"{probability:.2f}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
