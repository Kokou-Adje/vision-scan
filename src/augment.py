"""Stage 1: synthetic augmentation. Each image gets `multiplier` augmented copies."""

import argparse
import random
import shutil
from pathlib import Path

import albumentations as A
import cv2
from tqdm import tqdm


def build_transforms():
    bbp = A.BboxParams(format="yolo", label_fields=["cls"])
    return [
        ("rot", A.Compose([A.Rotate(limit=15, p=1.0, border_mode=cv2.BORDER_CONSTANT)], bbox_params=bbp)),
        ("hflip", A.Compose([A.HorizontalFlip(p=1.0)], bbox_params=bbp)),
        ("vflip", A.Compose([A.VerticalFlip(p=1.0)], bbox_params=bbp)),
        ("bright", A.Compose([A.RandomBrightnessContrast(0.3, 0.3, p=1.0)], bbox_params=bbp)),
        ("noise", A.Compose([A.GaussNoise(var_limit=(10, 50), p=1.0)], bbox_params=bbp)),
        ("blur", A.Compose([A.GaussianBlur(blur_limit=(3, 7), p=1.0)], bbox_params=bbp)),
    ]


def read_labels(path):
    if not path.exists():
        return [], []
    bboxes, classes = [], []
    for line in path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        c, cx, cy, w, h = parts
        bboxes.append([float(cx), float(cy), float(w), float(h)])
        classes.append(int(c))
    return bboxes, classes


def write_labels(path, bboxes, classes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(f"{c} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}"
                              for c, b in zip(classes, bboxes)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--multiplier", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    out_img = args.output / "images"
    out_lbl = args.output / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    transforms = build_transforms()
    images = sorted(args.input.glob("*.jpg")) + sorted(args.input.glob("*.png"))

    n_orig, n_aug = 0, 0
    for img_path in tqdm(images, desc="Augmenting"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        lbl_path = args.labels / (img_path.stem + ".txt")
        bboxes, classes = read_labels(lbl_path)

        # Copy original
        shutil.copy(img_path, out_img / img_path.name)
        if lbl_path.exists():
            shutil.copy(lbl_path, out_lbl / lbl_path.name)
        n_orig += 1

        # Generate augmented versions
        for name, tfm in random.sample(transforms, min(args.multiplier, len(transforms))):
            try:
                result = tfm(image=img, bboxes=bboxes, cls=classes)
            except Exception as e:
                print(f"Skipping {img_path.name} / {name}: {e}")
                continue
            out_name = f"{img_path.stem}_{name}{img_path.suffix}"
            cv2.imwrite(str(out_img / out_name), result["image"])
            write_labels(out_lbl / (Path(out_name).stem + ".txt"),
                         result["bboxes"], result["cls"])
            n_aug += 1

    print(f"Originals: {n_orig}, augmented: {n_aug}, total: {n_orig + n_aug}")


if __name__ == "__main__":
    main()
