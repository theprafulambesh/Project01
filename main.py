from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np

app = FastAPI(title="Signal AMC API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASSES = ['BPSK', 'QPSK', '8PSK', '16QAM', 'PAM4', 'GFSK']
session = ort.InferenceSession("rf_classifier.onnx")

class SignalPayload(BaseModel):
    signal: list[list[float]]

@app.get("/")
def health():
    return {"status": "online", "model": "rf_classifier.onnx"}

@app.post("/predict")
def predict(payload: SignalPayload):
    data = np.array(payload.signal, dtype=np.float32)
    if data.shape != (2, 128):
        raise HTTPException(status_code=400, detail=f"Expected shape (2, 128), got {data.shape}")
    
    # Power normalization
    norm = np.sqrt(np.mean(data[0]**2 + data[1]**2)) + 1e-8
    data = data / norm
    input_tensor = np.expand_dims(data, axis=0)
    
    outputs = session.run(None, {"iq_input": input_tensor})
    logits = outputs[0][0]
    
    exp_logits = np.exp(logits - np.max(logits))
    probs = (exp_logits / np.sum(exp_logits)).tolist()
    
    ranked = sorted(
        [{"modulation": cls, "confidence": round(float(p) * 100, 2)} for cls, p in zip(CLASSES, probs)],
        key=lambda x: x["confidence"],
        reverse=True
    )
    
    return {
        "top_prediction": ranked[0]["modulation"],
        "confidence": ranked[0]["confidence"],
        "rankings": ranked
    }