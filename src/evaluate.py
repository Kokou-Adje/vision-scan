"""Evaluate trained model on test split. Writes JSON for the report tables."""

import argparse
import json
from pathlib import Path
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, default=Path("results/metrics.json"))
    args = ap.parse_args()

    model = YOLO(str(args.weights))
    m = model.val(data=str(args.data), split="test")

    results = {
        "mAP50": float(m.box.map50),
        "mAP50-95": float(m.box.map),
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
        "per_class": {},
    }

    names = m.names if hasattr(m, "names") else {}
    if hasattr(m.box, "p") and m.box.p is not None:
        for i, cls_name in names.items():
            try:
                results["per_class"][cls_name] = {
                    "precision": float(m.box.p[i]),
                    "recall": float(m.box.r[i]),
                    "mAP50": float(m.box.ap50[i]),
                }
            except (IndexError, TypeError):
                continue

    print(f"\nmAP@0.5      = {results['mAP50']:.4f}")
    print(f"mAP@0.5:0.95 = {results['mAP50-95']:.4f}")
    print(f"Precision    = {results['precision']:.4f}")
    print(f"Recall       = {results['recall']:.4f}")
    print("\nPer-class:")
    for c, v in results["per_class"].items():
        print(f"  {c:12s}  P={v['precision']:.3f}  R={v['recall']:.3f}  mAP={v['mAP50']:.3f}")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {args.out_json}")


if __name__ == "__main__":
    main()
