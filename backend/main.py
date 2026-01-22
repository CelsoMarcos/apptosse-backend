from fastapi import FastAPI, File, UploadFile
import shutil
import uuid
import os
import joblib
import numpy as np

from backend.utils.audio_features import extract_features

from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="Cough Detection API")

MODEL_PATH = "backend/models/cough_model_svm.pkl"
SCALER_PATH = "backend/models/scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # em produção isto seria restrito
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Cough detection API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # guardar ficheiro temporário
    temp_filename = f"temp_{uuid.uuid4()}.wav"

    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # extrair features
    features = extract_features(temp_filename)
    features = scaler.transform(features)

    # prever
    probs = model.predict_proba(features)[0]
    prediction = int(np.argmax(probs))

    label = "Problema Respiratório" if prediction == 1 else "Normal"

    os.remove(temp_filename)

    return {
        "prediction": label,
        "probability": float(probs[prediction]),
        "disclaimer": "Resultado experimental. Não substitui diagnóstico médico."
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
