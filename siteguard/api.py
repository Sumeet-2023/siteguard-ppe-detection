import shutil, tempfile, uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from ultralytics import YOLO

from .video import process_video

WEIGHTS = "models/best.onnx"
app = FastAPI(title="SiteGuard PPE Detection")
model = YOLO(WEIGHTS)          # load once at startup, not per request


@app.get("/health")
def health():
    return {"status": "ok", "model": WEIGHTS}


@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode image")

    res = model.predict(img, verbose=False)[0]
    dets = [{"class": res.names[int(c)],
             "confidence": round(float(cf), 4),
             "bbox_xyxy": [round(float(v), 1) for v in box]}
            for box, c, cf in zip(res.boxes.xyxy.cpu().numpy(),
                                  res.boxes.cls.cpu().numpy(),
                                  res.boxes.conf.cpu().numpy())]

    return {"detections": dets,
            "people": len(dets),
            "violations": sum(d["class"] == "head" for d in dets)}


@app.post("/detect/video")
async def detect_video(file: UploadFile = File(...)):
    tmp = Path(tempfile.gettempdir()) / f"{uuid.uuid4()}{Path(file.filename).suffix}"
    with tmp.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)
    try:
        return process_video(str(tmp), WEIGHTS)
    finally:
        tmp.unlink(missing_ok=True)
