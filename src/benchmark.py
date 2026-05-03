"""Measure inference latency and FPS."""

import argparse
import time
from pathlib import Path
import numpy as np
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    model = YOLO(str(args.weights), task="detect")
    dummy = np.random.randint(0, 255, (args.imgsz, args.imgsz, 3), dtype=np.uint8)

    for _ in range(args.warmup):
        model.predict(dummy, imgsz=args.imgsz, device=args.device, verbose=False)

    times = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        model.predict(dummy, imgsz=args.imgsz, device=args.device, verbose=False)
        times.append(time.perf_counter() - t0)

    times = np.array(times)
    print(f"\nWeights:   {args.weights}")
    print(f"Latency:   {times.mean()*1000:.2f} ms (mean), {np.percentile(times, 95)*1000:.2f} ms (p95)")
    print(f"FPS:       {1.0/times.mean():.1f}")


if __name__ == "__main__":
    main()
