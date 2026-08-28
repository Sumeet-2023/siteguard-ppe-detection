"""Flatten the vodan37/yolo-helmethead Kaggle mirror into a single image/label
pool and remap its class indices to our convention.

This mirror ships pre-converted to YOLO format (not VOC XML as originally
assumed) and pre-split into train/valid/test -- but that split is a naive
random split over a dataset known to contain near-duplicates, which is
exactly the leakage Phase 2's honest pHash split exists to catch. So we
undo their split here and let make_splits.py + apply_splits.py redo it
group-disjointly.

Class remap: the mirror's helm.yaml declares names: ['head', 'helmet']
(0=head, 1=helmet). Our configs/data_shwd.yaml declares 0=helmet, 1=head,
so indices are swapped here to match.
"""
import argparse
from pathlib import Path

MIRROR_TO_OURS = {0: 1, 1: 0}  # mirror head->1, mirror helmet->0


def remap_label(src: Path, dst: Path) -> None:
    lines_out = []
    for line in src.read_text().splitlines():
        if not line.strip():
            continue
        cls_id, *coords = line.split()
        new_id = MIRROR_TO_OURS[int(cls_id)]
        lines_out.append(f"{new_id} {' '.join(coords)}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines_out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mirror-root", type=Path, required=True,
                     help="path to .../helm/helm (contains images/, labels/)")
    ap.add_argument("--out-root", type=Path, default=Path("data/shwd_raw"))
    a = ap.parse_args()

    n_images = n_boxes = 0
    for split in ["train", "valid", "test"]:
        img_dir = a.mirror_root / "images" / split
        label_dir = a.mirror_root / "labels" / split
        for img in sorted(img_dir.glob("*.jpg")):
            label_src = label_dir / f"{img.stem}.txt"
            if not label_src.exists():
                continue
            dst_img = a.out_root / "images" / img.name
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            if not dst_img.exists() and not dst_img.is_symlink():
                dst_img.symlink_to(img.resolve())
            dst_label = a.out_root / "labels" / f"{img.stem}.txt"
            remap_label(label_src, dst_label)
            n_images += 1
            n_boxes += len(dst_label.read_text().splitlines())

    print(f"Combined {n_images} images ({n_boxes} boxes) from train/valid/test "
          f"into {a.out_root} with classes remapped to 0=helmet, 1=head")


if __name__ == "__main__":
    main()
