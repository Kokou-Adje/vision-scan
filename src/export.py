"""Export to ONNX for portable deployment."""

import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    model = YOLO(str(args.weights))
    out = model.export(format="onnx", imgsz=args.imgsz, simplify=True, opset=12)
    print(f"Exported to: {out}")


if __name__ == "__main__":
    main()
