import cv2
from ultralytics import YOLO

from .tracker import ViolationMonitor


def process_video(src: str, weights: str, out_path: str | None = None) -> dict:
    model = YOLO(weights)

    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()

    writer = (cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
              if out_path else None)
    monitor = ViolationMonitor(fps=fps)

    # stream=True is essential -- without it Ultralytics buffers every frame in RAM.
    for idx, res in enumerate(model.track(source=src, tracker="bytetrack.yaml",
                                          persist=True, stream=True, verbose=False)):
        if res.boxes.id is not None:
            for tid, cid in zip(res.boxes.id.int().cpu().tolist(),
                                res.boxes.cls.int().cpu().tolist()):
                monitor.update(tid, model.names[cid], idx)
        if writer:
            writer.write(res.plot())

    if writer:
        writer.release()
    return {"summary": monitor.summary(), "events": monitor.events}
