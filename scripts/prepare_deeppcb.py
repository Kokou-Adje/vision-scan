"""Convert DeepPCB to YOLO format. Handles _not/ sibling annotation folders."""

import argparse
import random
import shutil
from pathlib import Path

import cv2

CLASS_REMAP = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}


def find_annotation(img_path):
    """DeepPCB annotation lives at ../{group}_not/{stem_without_test}.txt"""
    parent = img_path.parent              # e.g. .../group20085/20085
    grandparent = parent.parent           # e.g. .../group20085
    not_folder = grandparent / (parent.name + "_not")
    stem = img_path.stem.replace("_test", "")
    return not_folder / (stem + ".txt")


def convert_annotation(src_txt, dst_txt, img_w, img_h):
    if not src_txt.exists():
        return 0
    lines_out = []
    for line in src_txt.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            x1, y1, x2, y2, cls_id = map(int, parts)
        except ValueError:
            continue
        if cls_id not in CLASS_REMAP:
            continue
        cx = ((x1 + x2) / 2.0) / img_w
        cy = ((y1 + y2) / 2.0) / img_h
        w = abs(x2 - x1) / img_w
        h = abs(y2 - y1) / img_h
        if w <= 0 or h <= 0:
            continue
        lines_out.append(f"{CLASS_REMAP[cls_id]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    dst_txt.parent.mkdir(parents=True, exist_ok=True)
    dst_txt.write_text("\n".join(lines_out))
    return len(lines_out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    test_imgs = sorted(args.source.rglob("*_test.jpg"))
    if not test_imgs:
        raise FileNotFoundError(f"No *_test.jpg under {args.source}")

    random.shuffle(test_imgs)
    n = len(test_imgs)
    n_test = int(n * 0.15)
    n_val = int(n * 0.15)
    splits = {
        "test": test_imgs[:n_test],
        "val": test_imgs[n_test:n_test + n_val],
        "train": test_imgs[n_test + n_val:],
    }

    for split, imgs in splits.items():
        boxes = 0
        for img_path in imgs:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            dst_img = args.output / "images" / split / img_path.name
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(img_path, dst_img)
            ann_src = find_annotation(img_path)
            dst_ann = args.output / "labels" / split / (img_path.stem + ".txt")
            boxes += convert_annotation(ann_src, dst_ann, w, h)
        print(f"[{split}] {len(imgs)} images, {boxes} boxes")

    print(f"Done. Output at {args.output}")


if __name__ == "__main__":
    main()