#!/usr/bin/env python3
"""Standalone YOLO training script (M5+ c).

Usage:
    python train_yolo.py              # generate synth data + train + export ONNX
    python train_yolo.py --skip-synth # skip synth generation (data/synth/ exists)
    python train_yolo.py --epochs 30  # override epoch count

Requires: pip install ultralytics  (not in default venv)
"""
import argparse
import sys
import time
from pathlib import Path

ROOT   = Path(__file__).resolve().parent           # cv/
SYNTH  = ROOT / "data" / "synth"
MODELS = ROOT / "models"
ONNX   = MODELS / "yolo_spot.onnx"
LOCK   = MODELS / ".yolo_training"

sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8n spot detector")
    parser.add_argument("--skip-synth", action="store_true",
                        help="Skip synthetic data generation (use existing data/synth/)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--n-synth", type=int, default=2000,
                        help="Number of synthetic images to generate")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics", file=sys.stderr)
        sys.exit(1)

    MODELS.mkdir(exist_ok=True)

    # Write lockfile so sidecar can report "training" status
    LOCK.write_text(str(time.time()))
    try:
        _run(args, YOLO)
    finally:
        if LOCK.exists():
            LOCK.unlink()


def _run(args, YOLO) -> None:
    from chromalog_cv.yolo import generate_synthetic_dataset

    # ── Step 1: Generate synthetic data ─────────────────────────────────────
    if not args.skip_synth:
        print(f"[train_yolo] Generating {args.n_synth} synthetic images → {SYNTH}")
        real_dir = ROOT.parent / "training_pictures"
        generate_synthetic_dataset(
            real_images_dir=str(real_dir),
            out_dir=str(SYNTH),
            n=args.n_synth,
        )
        print("[train_yolo] Synthetic data ready.")
    else:
        print(f"[train_yolo] --skip-synth: using existing {SYNTH}")

    dataset_yaml = SYNTH / "dataset.yaml"
    if not dataset_yaml.exists():
        print(f"ERROR: {dataset_yaml} not found. Run without --skip-synth first.", file=sys.stderr)
        raise SystemExit(1)

    # ── Step 2: Pre-train on synthetic data ──────────────────────────────────
    print(f"[train_yolo] Training YOLOv8n for {args.epochs} epochs on synthetic data (device=mps)…")
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(dataset_yaml),
        epochs=args.epochs,
        imgsz=640,
        device="mps",
        project=str(ROOT / "runs"),
        name="yolo_spot",
        exist_ok=True,
        verbose=False,
    )

    # ── Step 3: Optional fine-tune on real annotated data ────────────────────
    real_dataset = ROOT / "data" / "real" / "dataset.yaml"
    if real_dataset.exists():
        print(f"[train_yolo] Fine-tuning on real data ({real_dataset}) for 20 epochs…")
        model.train(
            data=str(real_dataset),
            epochs=20,
            imgsz=640,
            device="mps",
            lr0=0.001,
            project=str(ROOT / "runs"),
            name="yolo_spot_finetune",
            exist_ok=True,
            verbose=False,
        )

    # ── Step 4: Export to ONNX ───────────────────────────────────────────────
    print(f"[train_yolo] Exporting to ONNX → {ONNX}")
    model.export(format="onnx", opset=12, simplify=True)
    # ultralytics writes to runs/.../weights/best.onnx — move to models/
    import shutil
    candidates = list((ROOT / "runs").rglob("best.onnx"))
    if not candidates:
        raise RuntimeError("ONNX export produced no best.onnx file")
    shutil.copy2(str(candidates[-1]), str(ONNX))
    print(f"[train_yolo] Model saved to {ONNX}")

    # ── Step 5: Smoke-test with ONNX Runtime ─────────────────────────────────
    try:
        import onnxruntime as ort
        import numpy as np
        sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
        dummy = np.zeros((1, 3, 640, 640), dtype=np.float32)
        t0 = time.perf_counter()
        sess.run(None, {sess.get_inputs()[0].name: dummy})
        ms = (time.perf_counter() - t0) * 1000
        print(f"[train_yolo] ONNX smoke-test passed. Inference latency: {ms:.1f} ms")
    except Exception as e:
        print(f"[train_yolo] WARNING: ONNX smoke-test failed: {e}", file=sys.stderr)

    print("[train_yolo] Done.")


if __name__ == "__main__":
    main()
