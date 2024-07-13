from fastapi import FastAPI, File, UploadFile, HTTPException
from google.cloud import aiplatform
from google.cloud.aiplatform.gapic.schema import predict
from google.protobuf import json_format
import base64
import io

app = FastAPI()

# Configuración del proyecto y endpoint
# PROJECT_ID = "284071077738"
PROJECT_ID = "neumdetect-428422"
ENDPOINT_ID = "7249016390252756992"
LOCATION = "us-central1"

# Inicializar el cliente de Vertex AI
aiplatform.init(project=PROJECT_ID, location=LOCATION)

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Leer el contenido del archivo
        contents = await file.read()

        # Codificar la imagen en base64
        encoded_content = base64.b64encode(contents).decode("utf-8")

        # Crear la instancia para la predicción
        instance = predict.instance.ImageClassificationPredictionInstance(
            content=encoded_content,
        ).to_value()

        # Obtener el endpoint
        endpoint = aiplatform.Endpoint(ENDPOINT_ID)

        # Realizar la predicción
        prediction = endpoint.predict(instances=[instance])

        # Procesar el resultado
        result = json_format.MessageToDict(prediction._pb)
        predictions = result['predictions'][0]['confidences']

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
