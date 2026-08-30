"""Remap SH17 (17-class YOLO format) labels down to our 2-class scheme
(helmet / head) so a model trained on SHWD can be evaluated zero-shot on it.

SH17 class order, per the dataset maintainer's own sh17.yaml
(github.com/ahmadmughees/SH17dataset/blob/master/sh17.yaml) -- verified
against instance counts on the actual Kaggle download (helmet=927,
head=11985, totaling 75,994 instances across all 17 classes, matching the
paper exactly):
  0 person, 1 ear, 2 ear-mufs, 3 face, 4 face-guard, 5 face-mask, 6 foot,
  7 tool, 8 glasses, 9 gloves, 10 helmet, 11 hands, 12 head,
  13 medical-suit, 14 shoes, 15 safety-suit, 16 safety-vest

Only 'helmet' and 'head' carry semantics comparable to our task; every other
box is dropped. This dataset ships flat (images/, labels/, no train/val/test
split) since Phase 6 uses the entire thing as a zero-shot test set -- no
fine-tuning, so no split is needed.
"""
import argparse
from pathlib import Path

SH17_NAMES = {
    0: "person", 1: "ear", 2: "ear-mufs", 3: "face", 4: "face-guard",
    5: "face-mask", 6: "foot", 7: "tool", 8: "glasses", 9: "gloves",
    10: "helmet", 11: "hands", 12: "head", 13: "medical-suit",
    14: "shoes", 15: "safety-suit", 16: "safety-vest",
}
# SH17 name -> our class id (0=helmet, 1=head), matching configs/data_shwd.yaml.
TO_OURS = {"helmet": 0, "head": 1}

IMG_EXTS = {".jpg", ".jpeg", ".png"}


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
    ap.add_argument("--images-dir", type=Path, required=True)
    ap.add_argument("--labels-dir", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, default=Path("data/sh17_remapped"))
    a = ap.parse_args()

    total_kept = total_dropped = n_images = n_no_relevant_boxes = 0
    images = sorted(p for p in a.images_dir.iterdir() if p.suffix.lower() in IMG_EXTS)

    for img in images:
        label_src = a.labels_dir / f"{img.stem}.txt"
        if not label_src.exists():
            continue

        dst_label = a.out_root / "labels" / "test" / f"{img.stem}.txt"
        kept, dropped = remap_label_file(label_src, dst_label)
        if kept == 0:
            n_no_relevant_boxes += 1

        dst_img = a.out_root / "images" / "test" / img.name
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        if not dst_img.exists() and not dst_img.is_symlink():
            dst_img.symlink_to(img.resolve())

        total_kept += kept
        total_dropped += dropped
        n_images += 1

    print(f"{n_images} images: {total_kept} boxes remapped to helmet/head, "
          f"{total_dropped} boxes dropped (other SH17 classes), "
          f"{n_no_relevant_boxes} images end up with zero helmet/head boxes")

    yaml_path = a.out_root / "data_sh17.yaml"
    yaml_path.write_text(
        f"path: {a.out_root.resolve()}\n"
        "train: images/test\nval: images/test\ntest: images/test\n"
        "names:\n  0: helmet\n  1: head\n"
    )
    print(f"Wrote {yaml_path}")


if __name__ == "__main__":
    main()
