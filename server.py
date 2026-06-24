import io
import os
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

from PIL import Image

# --------------------------------------------------
# Paths
# --------------------------------------------------

YOLOV5_DIR = Path(__file__).parent / "yolov5"
WEIGHTS_PATH = Path(__file__).parent / "best.pt"

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# FastAPI App
# --------------------------------------------------

app = FastAPI(title="SafeCityAI Helmet Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Load YOLO Model
# --------------------------------------------------

try:
    model = torch.hub.load(
        str(YOLOV5_DIR),
        "custom",
        path=str(WEIGHTS_PATH),
        source="local",
        force_reload=False,
        verbose=False,
    )

    model.conf = 0.25
    model.eval()

    print("✅ Model loaded successfully")
    print("Classes:", model.names)

except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# --------------------------------------------------
# Home Route
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "status": "SafeCityAI Detection API Running",
        "model_loaded": model is not None
    }

# --------------------------------------------------
# JSON Detection Endpoint
# --------------------------------------------------

@app.post("/detect")
async def detect(file: UploadFile = File(...)):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded."
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Empty file uploaded."
        )

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )

    results = model(image)

    print("\n===== RAW PREDICTIONS =====")
    print(results.xyxy[0])

    predictions = results.xyxy[0]

    detections = []

    for *box, conf, cls in predictions.tolist():

        x1, y1, x2, y2 = box

        detections.append({
            "class": model.names[int(cls)],
            "confidence": round(float(conf), 2),
            "xmin": round(x1, 1),
            "ymin": round(y1, 1),
            "xmax": round(x2, 1),
            "ymax": round(y2, 1)
        })

    return JSONResponse(
        content={
            "total_detections": len(detections),
            "detections": detections
        }
    )

# --------------------------------------------------
# Detection Image Endpoint
# --------------------------------------------------

@app.post("/detect-image")
async def detect_image(file: UploadFile = File(...)):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded."
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Empty file uploaded."
        )

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )

    results = model(image)

    img = np.array(image)

    for *box, conf, cls in results.xyxy[0].tolist():

        x1, y1, x2, y2 = map(int, box)

        label = (
            f"{model.names[int(cls)]} "
            f"{float(conf):.2f}"
        )

        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            img,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    filename = f"{uuid.uuid4().hex}.jpg"

    output_path = OUTPUT_DIR / filename

    success = cv2.imwrite(
    str(output_path),
    cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
)

    print("Save status:", success)
    print("Output path:", output_path)

    return FileResponse(
        path=str(output_path),
        media_type="image/jpeg",
        filename=filename
    )

# --------------------------------------------------
# Run Server
# --------------------------------------------------

if __name__ == "__main__":

    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )