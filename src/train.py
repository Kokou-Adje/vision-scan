"""Stage 2: fine-tune YOLOv8 on augmented PCB data."""

import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="vision_scan")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    yolo = YOLO(args.model)
    yolo.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        device=args.device,
        patience=30,
        optimizer="AdamW",
        lr0=0.001,
        cos_lr=True,
        plots=True,
        save=True,
    )

    metrics = yolo.val(data=str(args.data), split="test", name=f"{args.name}_test")
    print("=" * 60)
    print(f"mAP@0.5      = {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95 = {metrics.box.map:.4f}")
    print(f"Precision    = {metrics.box.mp:.4f}")
    print(f"Recall       = {metrics.box.mr:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
