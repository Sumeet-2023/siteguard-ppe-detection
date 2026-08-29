import argparse, json, time
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO


def latency(model, imgsz: int, device: str, warmup=20, runs=100) -> dict:
    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(warmup):                      # never benchmark a cold model
        model.predict(dummy, imgsz=imgsz, device=device, verbose=False)

    times = []
    for _ in range(runs):
        if device != "cpu":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.predict(dummy, imgsz=imgsz, device=device, verbose=False)
        if device != "cpu":
            torch.cuda.synchronize()             # without this, GPU numbers are fiction
        times.append((time.perf_counter() - t0) * 1000)

    t = np.array(times)
    return {"p50_ms": round(float(np.percentile(t, 50)), 2),
            "fps": round(1000 / float(t.mean()), 1)}


def evaluate(weights: Path, data: str, imgsz: int, device: str) -> dict:
    has_gpu = torch.cuda.is_available()
    eval_device = device if has_gpu else "cpu"

    model = YOLO(str(weights))
    m = model.val(data=data, split="test", imgsz=imgsz, device=eval_device, verbose=False)

    row = {"model": weights.parent.parent.name,
           "size_MB": round(weights.stat().st_size / 1e6, 1),
           "mAP50": round(float(m.box.map50), 4),
           "mAP50_95": round(float(m.box.map), 4)}
    for i, ap in enumerate(m.box.ap50):
        row[f"AP50_{model.names[i]}"] = round(float(ap), 4)

    if has_gpu:
        row |= {f"gpu_{k}": v for k, v in latency(model, imgsz, device).items()}
    row |= {f"cpu_{k}": v for k, v in latency(model, imgsz, "cpu").items()}
    return row


def to_md(rows: list[dict]) -> str:
    cols = list(rows[0])
    return "\n".join([
        "| " + " | ".join(cols) + " |",
        "|" + "|".join("---" for _ in cols) + "|",
        *["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows],
    ])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=Path, default=Path("runs/ppe"))
    ap.add_argument("--data", default="configs/data_shwd.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()

    rows = [evaluate(w, a.data, a.imgsz, a.device)
            for w in sorted(a.runs_dir.glob("*/weights/best.pt"))]
    Path("reports").mkdir(exist_ok=True)
    Path("reports/benchmark.json").write_text(json.dumps(rows, indent=2))
    Path("reports/benchmark.md").write_text(to_md(rows))
    print(to_md(rows))
