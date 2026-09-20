import os
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="AMC Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASSES = ['BPSK', 'QPSK', '8PSK', '16QAM', 'PAM4', 'GFSK']

# Model load with error handling
session = None
model_path = os.path.join(os.path.dirname(__file__), "rf_classifier.onnx")

try:
    import onnxruntime as ort
    if os.path.exists(model_path):
        session = ort.InferenceSession(model_path)
        print("ONNX model loaded successfully.")
    else:
        print(f"Warning: {model_path} not found.")
except Exception as e:
    print(f"Model load error: {e}")

class SignalPayload(BaseModel):
    signal: list[list[float]]

@app.get("/")
def health():
    return {
        "status": "online",
        "model_loaded": session is not None
    }

@app.post("/predict")
def predict(payload: SignalPayload):
    if session is None:
        raise HTTPException(status_code=500, detail="ONNX model session not initialized.")

    data = np.array(payload.signal, dtype=np.float32)
    if data.shape != (2, 128):
        raise HTTPException(status_code=400, detail=f"Expected shape (2, 128), got {data.shape}")
    
    # Unit energy normalization
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

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
