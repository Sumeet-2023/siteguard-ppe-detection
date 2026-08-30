"""Turn model draft predictions + manually-reviewed corrections into final
YOLO label files for the own_photos test set.

Two-step process (see reports/own_photos/README.md for why):
  1. Run yolo11s over data/own_photos/raw/*.jpg to get draft boxes
     (this script does that itself, saving reports/own_photos/draft_preds.json).
  2. Apply reports/own_photos/corrections.json -- a human-reviewed map of which
     draft boxes are real and which classes needed fixing -- to produce the
     final labels. corrections.json is committed and must be hand-edited if
     re-run against new/different source images.
"""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def run_draft_predictions(weights: Path, raw_dir: Path, conf: float) -> dict:
    model = YOLO(str(weights))
    results = {}
    for img_path in sorted(raw_dir.glob("*.jpg")):
        res = model.predict(str(img_path), imgsz=640, conf=conf, verbose=False)[0]
        dets = [
            {"cls": int(c), "name": model.names[int(c)], "conf": round(float(cf), 3),
             "xyxyn": [round(float(v), 4) for v in box]}
            for box, c, cf in zip(res.boxes.xyxyn.cpu().numpy(),
                                   res.boxes.cls.cpu().numpy(),
                                   res.boxes.conf.cpu().numpy())
        ]
        results[img_path.name] = dets
        print(f"{img_path.name}: {len(dets)} draft detections")
    return results


def apply_corrections(draft: dict, corrections: dict, out_root: Path, raw_dir: Path) -> int:
    img_dir = out_root / "images" / "test"
    label_dir = out_root / "labels" / "test"
    img_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    total_boxes = 0
    for fname, spec in corrections.items():
        if fname.startswith("_"):
            continue
        dets = draft[fname]
        keep, overrides = spec["keep"], {int(k): v for k, v in spec["class_overrides"].items()}
        lines = []
        for i in keep:
            det = dets[i]
            cls = overrides.get(i, det["cls"])
            x1, y1, x2, y2 = det["xyxyn"]
            xc, yc, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
            lines.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        (label_dir / fname.replace(".jpg", ".txt")).write_text("\n".join(lines))
        dst_img = img_dir / fname
        if not dst_img.exists() and not dst_img.is_symlink():
            dst_img.symlink_to((raw_dir / fname).resolve())
        total_boxes += len(lines)
        print(f"{fname}: {len(lines)} boxes")
    return total_boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=Path("runs/ppe/yolo11s/weights/best.pt"))
    ap.add_argument("--raw-dir", type=Path, default=Path("data/own_photos/raw"))
    ap.add_argument("--out-root", type=Path, default=Path("data/own_photos"))
    ap.add_argument("--corrections", type=Path,
                     default=Path("reports/own_photos/corrections.json"))
    ap.add_argument("--draft-out", type=Path,
                     default=Path("reports/own_photos/draft_preds.json"))
    ap.add_argument("--conf", type=float, default=0.15)
    a = ap.parse_args()

    draft = run_draft_predictions(a.weights, a.raw_dir, a.conf)
    a.draft_out.parent.mkdir(parents=True, exist_ok=True)
    a.draft_out.write_text(json.dumps(draft, indent=2))

    corrections = json.loads(a.corrections.read_text())
    total = apply_corrections(draft, corrections, a.out_root, a.raw_dir)

    yaml_path = a.out_root / "data_ownphotos.yaml"
    yaml_path.write_text(
        f"path: {a.out_root.resolve()}\n"
        "train: images/test\nval: images/test\ntest: images/test\n"
        "names:\n  0: helmet\n  1: head\n"
    )
    print(f"\n{total} total boxes. Wrote {yaml_path}")


if __name__ == "__main__":
    main()
