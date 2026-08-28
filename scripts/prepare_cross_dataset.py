"""Remap SH17 (17-class YOLO format) labels down to our 2-class scheme
(helmet / head) so a model trained on SHWD can be evaluated zero-shot on it.

SH17 class names, as published: person, ear, ear-mufs, face, face-guard,
face-mask, foot, tool, glasses, gloves, helmet, hands, head, medical-suit,
shoes, safety-suit, safety-vest. Verify this list against your actual
download's data.yaml -- ordering/spelling has varied between releases.

Only 'helmet' and 'head' carry semantics comparable to our task; every other
box is dropped. Write the mapping down explicitly (this file) rather than
burying it silently in eval code.
"""
import argparse
import shutil
from pathlib import Path

# SH17 class index -> name, per the dataset's published data.yaml.
SH17_NAMES = {
    0: "person", 1: "ear", 2: "ear-mufs", 3: "face", 4: "face-guard",
    5: "face-mask", 6: "foot", 7: "tool", 8: "glasses", 9: "gloves",
    10: "helmet", 11: "hands", 12: "head", 13: "medical-suit",
    14: "shoes", 15: "safety-suit", 16: "safety-vest",
}
# SH17 name -> our class id (0=helmet, 1=head). Everything else is dropped.
TO_OURS = {"helmet": 0, "head": 1}


def remap_label_file(src: Path, dst: Path) -> tuple[int, int]:
    kept, dropped = 0, 0
    lines_out = []
    for line in src.read_text().splitlines():
        if not line.strip():
            continue
        cls_id, *coords = line.split()
        name = SH17_NAMES.get(int(cls_id))
        if name not in TO_OURS:
            dropped += 1
            continue
        lines_out.append(f"{TO_OURS[name]} {' '.join(coords)}")
        kept += 1
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines_out))
    return kept, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sh17-root", type=Path, required=True,
                     help="root containing images/ and labels/")
    ap.add_argument("--out-root", type=Path, default=Path("data/sh17_remapped"))
    ap.add_argument("--split", default="test")
    a = ap.parse_args()

    img_dir = a.sh17_root / "images" / a.split
    label_dir = a.sh17_root / "labels" / a.split

    total_kept = total_dropped = n_images = 0
    for img in sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png")):
        label_src = label_dir / f"{img.stem}.txt"
        if not label_src.exists():
            continue
        kept, dropped = remap_label_file(
            label_src, a.out_root / "labels" / a.split / f"{img.stem}.txt")
        dst_img = a.out_root / "images" / a.split / img.name
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        if not dst_img.exists():
            shutil.copy(img, dst_img)
        total_kept += kept
        total_dropped += dropped
        n_images += 1

    print(f"{n_images} images: {total_kept} boxes remapped to helmet/head, "
          f"{total_dropped} boxes dropped (other SH17 classes)")

    yaml_path = a.out_root / "data_sh17.yaml"
    yaml_path.write_text(
        f"path: {a.out_root.resolve()}\n"
        f"train: images/{a.split}\nval: images/{a.split}\ntest: images/{a.split}\n"
        "names:\n  0: helmet\n  1: head\n"
    )
    print(f"Wrote {yaml_path}")


if __name__ == "__main__":
    main()
