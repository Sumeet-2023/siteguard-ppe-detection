"""Bucket images by perceptual hash, then split group-disjointly."""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import imagehash
from PIL import Image
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("configs/splits.json"))
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    paths = sorted(p for p in args.img_dir.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    groups = defaultdict(list)
    for p in tqdm(paths, desc="hashing"):
        with Image.open(p) as im:
            groups[str(imagehash.phash(im.convert("RGB")))].append(p.name)

    dupes = sum(len(v) - 1 for v in groups.values())
    print(f"{len(paths)} images -> {len(groups)} groups "
          f"({dupes} duplicates, {100 * dupes / len(paths):.1f}%)")

    # Shuffle groups, not images. That is the whole point.
    keys = list(groups)
    random.Random(args.seed).shuffle(keys)

    n_train, n_val = int(0.70 * len(paths)), int(0.15 * len(paths))
    splits, count = {"train": [], "val": [], "test": []}, 0
    for k in keys:
        bucket = ("train" if count < n_train
                  else "val" if count < n_train + n_val else "test")
        splits[bucket].extend(groups[k])
        count += len(groups[k])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(splits, indent=2))
    for k, v in splits.items():
        print(f"  {k}: {len(v)}")


if __name__ == "__main__":
    main()
