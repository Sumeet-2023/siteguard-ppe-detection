"""Convert Pascal VOC XML annotations to YOLO txt format."""
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

# SHWD uses "hat" for a helmeted head and "person" for a bare head.
# Verify against your actual download -- class names vary between mirrors.
CLASS_MAP = {"hat": 0, "helmet": 0, "person": 1, "head": 1}


def convert_one(xml_path: Path, out_dir: Path) -> tuple[int, int]:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    img_w, img_h = int(size.find("width").text), int(size.find("height").text)
    if img_w <= 0 or img_h <= 0:
        return 0, 1

    lines, skipped = [], 0
    for obj in root.iter("object"):
        name = obj.find("name").text.strip().lower()
        if name not in CLASS_MAP:
            skipped += 1
            continue

        bb = obj.find("bndbox")
        x1 = max(0.0, float(bb.find("xmin").text))
        y1 = max(0.0, float(bb.find("ymin").text))
        x2 = min(float(img_w), float(bb.find("xmax").text))
        y2 = min(float(img_h), float(bb.find("ymax").text))

        # VOC boxes routinely exceed image bounds; drop anything degenerate.
        if x2 - x1 < 4 or y2 - y1 < 4:
            skipped += 1
            continue

        xc, yc = ((x1 + x2) / 2) / img_w, ((y1 + y2) / 2) / img_h
        w, h = (x2 - x1) / img_w, (y2 - y1) / img_h
        lines.append(f"{CLASS_MAP[name]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{xml_path.stem}.txt").write_text("\n".join(lines))
    return len(lines), skipped


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ann-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()

    kept = dropped = 0
    xmls = sorted(a.ann_dir.glob("*.xml"))
    for x in xmls:
        k, d = convert_one(x, a.out_dir)
        kept += k
        dropped += d
    print(f"{len(xmls)} files -> {kept} boxes kept, {dropped} dropped")
