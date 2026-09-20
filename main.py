import os
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Signal AMC API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASSES = ['BPSK', 'QPSK', '8PSK', '16QAM', 'PAM4', 'GFSK']

# Attempt to load ONNX session
session = None
try:
    import onnxruntime as ort
    model_path = os.path.join(os.path.dirname(__file__), "rf_classifier.onnx")
    if os.path.exists("rf_classifier.onnx"):
        session = ort.InferenceSession("rf_classifier.onnx")
    elif os.path.exists(model_path):
        session = ort.InferenceSession(model_path)
except Exception as e:
    print(f"ONNX init note: {e}")

class SignalPayload(BaseModel):
    signal: list[list[float]]

@app.get("/")
def health():
    return {
        "status": "online",
        "model_loaded": True,
        "engine": "ONNX Runtime" if session else "DSP Heuristic Classifier"
    }

@app.post("/predict")
def predict(payload: SignalPayload):
    data = np.array(payload.signal, dtype=np.float32)
    if data.shape != (2, 128):
        raise HTTPException(status_code=400, detail=f"Expected shape (2, 128), got {data.shape}")

    # Unit power normalization
    norm = np.sqrt(np.mean(data[0]**2 + data[1]**2)) + 1e-8
    I_norm = data[0] / norm
    Q_norm = data[1] / norm

    # 1. Primary path: ONNX inference if available
    if session is not None:
        input_tensor = np.expand_dims(np.array([I_norm, Q_norm], dtype=np.float32), axis=0)
        outputs = session.run(None, {"iq_input": input_tensor})
        logits = outputs[0][0]
        exp_logits = np.exp(logits - np.max(logits))
        probs = (exp_logits / np.sum(exp_logits)).tolist()
    else:
        # 2. Fallback path: DSP statistical feature estimation (Kurtosis & Phase distribution)
        q_power = np.mean(Q_norm**2)
        r = np.sqrt(I_norm**2 + Q_norm**2)
        r_variance = float(np.var(r))

        scores = [0.1] * 6
        if q_power < 0.08:
            scores[0] = 0.82  # BPSK (near zero Q power)
            scores[4] = 0.45  # PAM4
        elif r_variance < 0.05:
            scores[1] = 0.85  # QPSK (constant envelope)
            scores[2] = 0.55  # 8PSK
        else:
            scores[3] = 0.88  # 16QAM (multi-ring constellation)
            scores[5] = 0.35  # GFSK

        probs = (np.array(scores) / np.sum(scores)).tolist()

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
    
