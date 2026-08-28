"""Materialise configs/splits.json into the images/{split}/ and labels/{split}/
layout that data_shwd.yaml expects. Symlinks by default so re-running is cheap
and the dataset is never duplicated on disk.
"""
import argparse
import json
from pathlib import Path


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        dst.write_bytes(src.read_bytes())
    else:
        dst.symlink_to(src.resolve())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=Path, default=Path("configs/splits.json"))
    ap.add_argument("--img-dir", type=Path, required=True)
    ap.add_argument("--label-dir", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--copy", action="store_true", help="copy instead of symlink")
    args = ap.parse_args()

    splits = json.loads(args.splits.read_text())
    for split, names in splits.items():
        missing_labels = 0
        for name in names:
            img_src = args.img_dir / name
            label_src = args.label_dir / f"{Path(name).stem}.txt"

            link_or_copy(img_src, args.out_root / "images" / split / name, args.copy)
            if label_src.exists():
                link_or_copy(label_src,
                             args.out_root / "labels" / split / f"{Path(name).stem}.txt",
                             args.copy)
            else:
                missing_labels += 1
        print(f"{split}: {len(names)} images, {missing_labels} missing labels")


if __name__ == "__main__":
    main()
