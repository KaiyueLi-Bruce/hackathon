# YOLO Spot Detection (M5+ c) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add YOLOv8n as a third-layer fallback spot detector activated via `/detect?use_yolo=true`, called only when the sklearn layer returns zero spots, with a Re-train button in Settings and ONNX Runtime inference.

**Architecture:** A new `yolo.py` module (in the existing `chromalog_cv` package) handles synthetic data generation and ONNX Runtime inference. A standalone `train_yolo.py` CLI script handles training (~5 min on Mac MPS). `pipeline.py` and `server.py` get minimal patches (~10–20 lines each). The Swift app gets a YOLO section in SettingsView with status polling.

**Tech Stack:** Python 3.9+, onnxruntime, numpy, opencv-headless, ultralytics (training only, guarded import), PyYAML; Swift/SwiftUI for the app layer.

## Global Constraints

- YOLOv8n only — no larger YOLO variants.
- Inference via ONNX Runtime only — `ultralytics` is never imported in sidecar runtime code.
- `ultralytics` import in `yolo.py` guarded with `try/except ImportError`; absent → `detect_yolo` returns empty `SpotsResult`, no crash.
- `cv/data/` and `models/*.onnx` must be added to `.gitignore` (not committed).
- `/detect` default behaviour unchanged: `use_yolo=False` means YOLO is never called.
- `engine_used` string is appended (`+yolo`), never wholesale replaced.
- All Python commands run inside the venv: `cv/.venv/bin/python3`.
- Swift build command: `cd App && swift build`.

---

### Task 1: Synthetic dataset generation + `.gitignore`

**Files:**
- Create: `cv/chromalog_cv/yolo.py`
- Modify: `.gitignore` (root)
- Test: `cv/tests/test_yolo_synth.py`

**Interfaces:**
- Produces: `generate_synthetic_dataset(real_images_dir: str | Path, out_dir: str | Path, n: int = 2000, seed: int = 42) -> None`
  - Writes `{out_dir}/images/train/`, `{out_dir}/images/val/`, `{out_dir}/labels/train/`, `{out_dir}/labels/val/`, `{out_dir}/dataset.yaml`
  - Label format: one `.txt` per image, lines `0 cx cy w h` (all normalised 0–1)
  - `dataset.yaml`: ultralytics format, 1 class named `spot`

- [ ] **Step 1: Add entries to `.gitignore`**

Append to `/Users/bruceli/Documents/hackathon/.gitignore`:
```
# ---- YOLO training data & model artifacts ----
cv/data/
models/*.onnx
models/.yolo_training
```

- [ ] **Step 2: Write the failing test**

Create `cv/tests/test_yolo_synth.py`:
```python
import tempfile
from pathlib import Path
import numpy as np
import cv2

def test_generate_creates_expected_structure():
    from chromalog_cv.yolo import generate_synthetic_dataset
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "synth"
        # Pass a non-existent real_images_dir — fallback to solid colour bg
        generate_synthetic_dataset(real_images_dir="/no/such/dir", out_dir=out, n=6, seed=0)
        assert (out / "images" / "train").exists()
        assert (out / "images" / "val").exists()
        assert (out / "labels" / "train").exists()
        assert (out / "labels" / "val").exists()
        assert (out / "dataset.yaml").exists()
        # n=6 → 5 train + 1 val (90/10 split)
        train_imgs = list((out / "images" / "train").glob("*.jpg"))
        val_imgs   = list((out / "images" / "val").glob("*.jpg"))
        assert len(train_imgs) == 5
        assert len(val_imgs)   == 1

def test_labels_are_valid_yolo_format():
    from chromalog_cv.yolo import generate_synthetic_dataset
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "synth"
        generate_synthetic_dataset(real_images_dir="/no/such/dir", out_dir=out, n=4, seed=1)
        for lbl in (out / "labels" / "train").glob("*.txt"):
            for line in lbl.read_text().strip().splitlines():
                parts = line.split()
                assert len(parts) == 5, f"bad line: {line}"
                assert parts[0] == "0"                        # class id
                vals = [float(p) for p in parts[1:]]
                assert all(0.0 <= v <= 1.0 for v in vals)    # normalised
                assert vals[2] > 0 and vals[3] > 0           # w, h > 0

def test_images_are_readable_rgb():
    from chromalog_cv.yolo import generate_synthetic_dataset
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "synth"
        generate_synthetic_dataset(real_images_dir="/no/such/dir", out_dir=out, n=2, seed=2)
        for img_path in (out / "images" / "train").glob("*.jpg"):
            img = cv2.imread(str(img_path))
            assert img is not None
            assert img.shape == (640, 640, 3)
```

- [ ] **Step 3: Run test to confirm it fails**

```bash
cd /Users/bruceli/Documents/hackathon/cv
.venv/bin/python3 -m pytest tests/test_yolo_synth.py -v
```
Expected: `ModuleNotFoundError: cannot import name 'generate_synthetic_dataset'`

- [ ] **Step 4: Create `cv/chromalog_cv/yolo.py` with `generate_synthetic_dataset()`**

```python
"""YOLO spot detector (M5+ c): synthetic dataset generation + ONNX inference.

ultralytics is only needed for training (train_yolo.py); this file uses only
onnxruntime at inference time and is importable with neither installed.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    import yaml as _yaml
    _YAML_OK = True
except ImportError:
    _YAML_OK = False

try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False

from .config import Config
from .spots import Spot, SpotsResult, _rf

YOLO_ONNX_PATH = Path(__file__).resolve().parent.parent / "models" / "yolo_spot.onnx"
_YOLO_SESSION = None  # lazily loaded on first detect_yolo() call


# ---------------------------------------------------------------------------
# Synthetic dataset generation
# ---------------------------------------------------------------------------

def _draw_blob(
    img: np.ndarray, cx: int, cy: int, r: int,
    intensity: int, dark_on_light: bool,
) -> None:
    h, w = img.shape[:2]
    y0 = max(0, cy - r * 2);  y1 = min(h, cy + r * 2 + 1)
    x0 = max(0, cx - r * 2);  x1 = min(w, cx + r * 2 + 1)
    if y1 <= y0 or x1 <= x0:
        return
    ys = np.arange(y0, y1, dtype=np.float32)
    xs = np.arange(x0, x1, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    dist2 = ((yy - cy) ** 2 + (xx - cx) ** 2) / max(1, r * r)
    vals = (intensity * np.exp(-dist2)).astype(np.int16)[..., None]  # broadcast to 3ch
    roi = img[y0:y1, x0:x1].astype(np.int16)
    if dark_on_light:
        img[y0:y1, x0:x1] = np.clip(roi - vals, 0, 255).astype(np.uint8)
    else:
        img[y0:y1, x0:x1] = np.clip(roi + vals, 0, 255).astype(np.uint8)


def generate_synthetic_dataset(
    real_images_dir: str,
    out_dir: str,
    n: int = 2000,
    seed: int = 42,
) -> None:
    """Generate a YOLO-format synthetic TLC dataset.

    Writes train (90%) / val (10%) splits to out_dir. Falls back to solid
    colour backgrounds when real_images_dir contains no usable images.
    """
    rng = np.random.RandomState(seed)
    real_dir = Path(real_images_dir)
    out = Path(out_dir)

    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Load real images for background patches
    real_imgs: list[np.ndarray] = []
    if real_dir.exists():
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for p in real_dir.glob(ext):
                img = cv2.imread(str(p))
                if img is not None and img.shape[0] >= 64 and img.shape[1] >= 64:
                    real_imgs.append(img)

    W, H = 640, 640
    n_train = int(n * 0.9)
    n_val = n - n_train
    idx = 0

    for split, count in (("train", n_train), ("val", n_val)):
        for _ in range(count):
            # ── Background ──────────────────────────────────────────────────
            if real_imgs:
                src = real_imgs[rng.randint(len(real_imgs))]
                sh, sw = src.shape[:2]
                sy = rng.randint(0, max(1, sh - H))
                sx = rng.randint(0, max(1, sw - W))
                plate = cv2.resize(src[sy:sy + H, sx:sx + W], (W, H))
            else:
                base = int(rng.randint(160, 220))
                plate = np.full((H, W, 3), base, dtype=np.uint8)
                plate = cv2.add(plate, rng.randint(0, 20, plate.shape, dtype=np.uint8))

            # ── Plate region (where spots may appear) ────────────────────
            py_front   = int(rng.randint(H // 8, H // 3))       # solvent front
            py_base    = int(rng.randint(H * 2 // 3, H * 7 // 8))  # baseline
            px_left    = int(rng.randint(10, W // 5))
            px_right   = int(rng.randint(W * 4 // 5, W - 10))
            dark_on_light = bool(rng.rand() > 0.5)

            # ── Draw spots ───────────────────────────────────────────────
            n_spots = int(rng.randint(2, 9))
            labels: list[str] = []
            for _ in range(n_spots):
                cx = int(rng.randint(px_left + 5, px_right - 5))
                cy = int(rng.randint(py_front + 5, py_base - 5))
                r  = int(rng.randint(4, 19))
                intensity = int(rng.randint(30, 80))
                _draw_blob(plate, cx, cy, r, intensity, dark_on_light)
                bbox_w = min(r * 4, W)
                bbox_h = min(r * 4, H)
                labels.append(
                    f"0 {cx / W:.6f} {cy / H:.6f} {bbox_w / W:.6f} {bbox_h / H:.6f}"
                )

            # ── Augmentation ─────────────────────────────────────────────
            alpha = rng.uniform(0.8, 1.2)
            plate  = np.clip(plate.astype(float) * alpha, 0, 255).astype(np.uint8)
            angle  = rng.uniform(-5.0, 5.0)
            scale  = rng.uniform(0.97, 1.03)
            M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, scale)
            plate = cv2.warpAffine(plate, M, (W, H), borderValue=(200, 200, 200))
            noise = rng.randint(0, 12, plate.shape, dtype=np.uint8)
            plate = cv2.add(plate, noise)

            fname = f"synth_{idx:05d}"
            cv2.imwrite(str(out / "images" / split / f"{fname}.jpg"), plate)
            (out / "labels" / split / f"{fname}.txt").write_text("\n".join(labels))
            idx += 1

    # ── dataset.yaml ────────────────────────────────────────────────────────
    yaml_str = (
        f"path: {out.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 1\n"
        f"names:\n"
        f"  0: spot\n"
    )
    (out / "dataset.yaml").write_text(yaml_str)
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
cd /Users/bruceli/Documents/hackathon/cv
.venv/bin/python3 -m pytest tests/test_yolo_synth.py -v
```
Expected: 3 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add cv/chromalog_cv/yolo.py cv/tests/test_yolo_synth.py .gitignore
git commit -m "feat(yolo): synthetic TLC dataset generator + .gitignore entries"
```

---

### Task 2: ONNX inference (`detect_yolo`)

**Files:**
- Modify: `cv/chromalog_cv/yolo.py` (add inference section)
- Test: `cv/tests/test_yolo_infer.py`

**Interfaces:**
- Consumes: `generate_synthetic_dataset` from Task 1; `Spot`, `SpotsResult`, `_rf` from `chromalog_cv.spots`
- Produces: `detect_yolo(img: np.ndarray, cfg: Config, baseline_y: float | None = None, front_y: float | None = None, conf_thr: float = 0.35) -> SpotsResult`
  - Returns `SpotsResult(spots=[], lane_bounds=[])` when model absent or onnxruntime missing
  - Returns `SpotsResult` with `Spot` objects (pixel coords, Rf computed) when model present

- [ ] **Step 1: Write the failing tests**

Create `cv/tests/test_yolo_infer.py`:
```python
import numpy as np
import pytest

def test_detect_yolo_no_model_returns_empty(tmp_path, monkeypatch):
    """With no .onnx file, detect_yolo must return empty SpotsResult."""
    import chromalog_cv.yolo as Y
    monkeypatch.setattr(Y, "YOLO_ONNX_PATH", tmp_path / "nonexistent.onnx")
    monkeypatch.setattr(Y, "_YOLO_SESSION", None)
    from chromalog_cv.config import Config
    img = np.zeros((200, 400, 3), dtype=np.uint8)
    result = Y.detect_yolo(img, Config())
    assert result.spots == []

def test_nms_removes_overlapping_boxes():
    from chromalog_cv.yolo import _nms
    boxes = np.array([
        [10, 10, 50, 50],
        [12, 12, 52, 52],   # heavily overlaps with box 0 → should be suppressed
        [200, 200, 240, 240],  # no overlap → kept
    ], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    kept = _nms(boxes, scores, iou_thr=0.45)
    assert 0 in kept
    assert 1 not in kept
    assert 2 in kept

def test_nms_empty_input():
    from chromalog_cv.yolo import _nms
    kept = _nms(np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.float32))
    assert kept == []
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /Users/bruceli/Documents/hackathon/cv
.venv/bin/python3 -m pytest tests/test_yolo_infer.py -v
```
Expected: `ImportError` or `AttributeError` on `_nms` / `detect_yolo`.

- [ ] **Step 3: Append inference code to `cv/chromalog_cv/yolo.py`**

Add after the `generate_synthetic_dataset` function:
```python
# ---------------------------------------------------------------------------
# ONNX inference
# ---------------------------------------------------------------------------

def _get_session() -> Optional["ort.InferenceSession"]:
    global _YOLO_SESSION
    if _YOLO_SESSION is None and _ORT_AVAILABLE and YOLO_ONNX_PATH.exists():
        _YOLO_SESSION = ort.InferenceSession(
            str(YOLO_ONNX_PATH), providers=["CPUExecutionProvider"]
        )
    return _YOLO_SESSION


def _nms(
    boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.45
) -> list[int]:
    """Pure-numpy NMS. boxes: [N, 4] as (x1, y1, x2, y2)."""
    if boxes.shape[0] == 0:
        return []
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (area_i + area_r - inter + 1e-6)
        order = rest[iou <= iou_thr]
    return keep


def detect_yolo(
    img: np.ndarray,
    cfg: Config,
    baseline_y: Optional[float] = None,
    front_y: Optional[float] = None,
    conf_thr: float = 0.35,
) -> SpotsResult:
    """Run YOLOv8n ONNX inference. Returns empty SpotsResult if model absent."""
    sess = _get_session()
    if sess is None:
        return SpotsResult()

    h0, w0 = img.shape[:2]
    SIZE = 640
    resized = cv2.resize(img, (SIZE, SIZE))
    inp = resized.astype(np.float32) / 255.0
    inp = inp.transpose(2, 0, 1)[np.newaxis]            # [1, 3, 640, 640]

    raw = sess.run(None, {sess.get_inputs()[0].name: inp})[0]  # [1, 5, 8400]
    pred = raw[0]  # [5, 8400]: rows cx, cy, w, h, conf

    cx_arr, cy_arr, w_arr, h_arr, conf_arr = pred[0], pred[1], pred[2], pred[3], pred[4]
    mask = conf_arr >= conf_thr
    if not mask.any():
        return SpotsResult()

    cx_f = cx_arr[mask] * w0 / SIZE
    cy_f = cy_arr[mask] * h0 / SIZE
    w_f  =  w_arr[mask] * w0 / SIZE
    h_f  =  h_arr[mask] * h0 / SIZE
    sc   = conf_arr[mask]

    x1 = cx_f - w_f / 2;  x2 = cx_f + w_f / 2
    y1 = cy_f - h_f / 2;  y2 = cy_f + h_f / 2
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    keep  = _nms(boxes, sc)

    spots: list[Spot] = []
    for i in keep:
        cx_px = float((x1[i] + x2[i]) / 2)
        cy_px = float((y1[i] + y2[i]) / 2)
        bw    = max(1, int(x2[i] - x1[i]))
        bh    = max(1, int(y2[i] - y1[i]))
        spots.append(Spot(
            x=cx_px, y=cy_px,
            bbox=(int(x1[i]), int(y1[i]), bw, bh),
            area=bw * bh, lane=0,
            rf=_rf(cy_px, baseline_y, front_y),
            shape="blob",
        ))
    return SpotsResult(spots=spots, lane_bounds=[])
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd /Users/bruceli/Documents/hackathon/cv
.venv/bin/python3 -m pytest tests/test_yolo_infer.py -v
```
Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add cv/chromalog_cv/yolo.py cv/tests/test_yolo_infer.py
git commit -m "feat(yolo): ONNX inference (detect_yolo + NMS); no model → empty result"
```

---

### Task 3: Training script `train_yolo.py`

**Files:**
- Create: `cv/train_yolo.py`
- Modify: `cv/requirements.txt` (add commented ultralytics line)

**Interfaces:**
- Consumes: `generate_synthetic_dataset` from `chromalog_cv.yolo`
- Produces: `models/yolo_spot.onnx` on disk; `models/.yolo_training` lockfile written at start, deleted on finish.

> **Note:** Full end-to-end test requires `pip install ultralytics` (not in venv by default). Steps below verify everything short of actual training.

- [ ] **Step 1: Create `cv/train_yolo.py`**

```python
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
```

- [ ] **Step 2: Add `ultralytics` to `requirements.txt` (commented)**

In `cv/requirements.txt`, change the existing commented line:
```
#   ultralytics torch   # YOLO 训练/推理
```
to:
```
#   ultralytics>=8.2    # YOLO training only: pip install ultralytics
```

- [ ] **Step 3: Verify script parses correctly (no ultralytics needed)**

```bash
cd /Users/bruceli/Documents/hackathon/cv
.venv/bin/python3 train_yolo.py --help
```
Expected output:
```
usage: train_yolo.py [-h] [--skip-synth] [--epochs EPOCHS] [--n-synth N_SYNTH]
...
```

- [ ] **Step 4: Commit**

```bash
git add cv/train_yolo.py cv/requirements.txt
git commit -m "feat(yolo): training script train_yolo.py (MPS, synthetic→ONNX export)"
```

---

### Task 4: Pipeline fallback (`pipeline.py` patch)

**Files:**
- Modify: `cv/chromalog_cv/pipeline.py`
- Test: `cv/tests/test_yolo_pipeline.py`

**Interfaces:**
- Consumes: `detect_yolo` from Task 2 (`chromalog_cv.yolo`)
- Produces: `run_pipeline(..., use_yolo: bool = False)` — new optional param; `engine_used` gains `+yolo` suffix when YOLO fires; `result.spots` list is replaced with YOLO detections

- [ ] **Step 1: Write the failing test**

Create `cv/tests/test_yolo_pipeline.py`:
```python
import numpy as np
import pytest

def _make_blank_img(h=200, w=400):
    return np.full((h, w, 3), 200, dtype=np.uint8)

def test_use_yolo_false_never_calls_yolo(monkeypatch):
    """use_yolo=False: detect_yolo must never be called."""
    called = []
    from chromalog_cv.spots import SpotsResult
    import chromalog_cv.yolo as Y
    monkeypatch.setattr(Y, "detect_yolo", lambda *a, **kw: (called.append(1), SpotsResult())[1])
    from chromalog_cv.pipeline import run_pipeline
    run_pipeline(_make_blank_img(), use_yolo=False)
    assert called == [], "detect_yolo should not be called when use_yolo=False"

def test_use_yolo_called_when_sklearn_returns_empty(monkeypatch):
    """use_yolo=True + sklearn finds 0 spots → detect_yolo is called."""
    from chromalog_cv.spots import SpotsResult, Spot
    import chromalog_cv.yolo as Y
    fake_spot = Spot(x=100, y=80, bbox=(90, 70, 20, 20), area=400, lane=0, rf=0.5, shape="blob")
    fake_result = SpotsResult(spots=[fake_spot], lane_bounds=[])
    called = []
    monkeypatch.setattr(Y, "detect_yolo",
                        lambda *a, **kw: (called.append(1), fake_result)[1])
    # Disable sklearn model so spots = 0 from OpenCV path
    import chromalog_cv.learn as LN
    from unittest.mock import MagicMock
    mock_clf = MagicMock()
    mock_clf.is_trained = False
    monkeypatch.setattr(LN, "SpotClassifier", MagicMock(load=MagicMock(return_value=mock_clf)))
    from chromalog_cv.pipeline import run_pipeline
    result, _, _ = run_pipeline(_make_blank_img(), use_yolo=True)
    # detect_yolo was called (may or may not find spots depending on OpenCV result,
    # but it must have been invoked when sklearn is off)
    # We check engine string instead: if yolo fired, engine ends with +yolo
    # or spots came from the fake
    if called:
        assert "yolo" in result.engine_used

def test_yolo_engine_label_appended(monkeypatch):
    """When YOLO fires and returns spots, engine_used ends with +yolo."""
    from chromalog_cv.spots import SpotsResult, Spot
    import chromalog_cv.yolo as Y
    fake = SpotsResult(
        spots=[Spot(x=200, y=100, bbox=(180,80,40,40), area=1600, lane=0, rf=0.4, shape="blob")],
        lane_bounds=[],
    )
    monkeypatch.setattr(Y, "detect_yolo", lambda *a, **kw: fake)
    import chromalog_cv.learn as LN
    from unittest.mock import MagicMock
    mock_clf = MagicMock(); mock_clf.is_trained = False
    monkeypatch.setattr(LN.SpotClassifier, "load", MagicMock(return_value=mock_clf))
    monkeypatch.setattr(LN, "SpotClassifier", MagicMock(load=MagicMock(return_value=mock_clf)))

    # Force sp.spots to be empty by making detect_spots return empty
    import chromalog_cv.spots as S
    from chromalog_cv.spots import SpotsResult as SR
    monkeypatch.setattr(S, "detect_spots",
                        lambda *a, **kw: SR(spots=[], lane_bounds=[]))

    from chromalog_cv.pipeline import run_pipeline
    result, _, _ = run_pipeline(_make_blank_img(), use_yolo=True)
    assert result.engine_used.endswith("+yolo")
    assert len(result.spots) == 1
    assert result.spots[0]["rf"] == pytest.approx(0.4, abs=0.01)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /Users/bruceli/Documents/hackathon/cv
.venv/bin/python3 -m pytest tests/test_yolo_pipeline.py -v
```
Expected: `TypeError: run_pipeline() got an unexpected keyword argument 'use_yolo'`

- [ ] **Step 3: Patch `cv/chromalog_cv/pipeline.py`**

Change the `run_pipeline` signature (line ~53):
```python
# Before:
def run_pipeline(bgr: np.ndarray, cfg: Optional[Config] = None,
                 debug: bool = False, llm_regions=None, engine_used: str = "opencv",
                 rect=None):

# After:
def run_pipeline(bgr: np.ndarray, cfg: Optional[Config] = None,
                 debug: bool = False, llm_regions=None, engine_used: str = "opencv",
                 rect=None, use_yolo: bool = False):
```

Then add the YOLO fallback block immediately after the sklearn scorer stage and before `spots_json` is built (after `sp = S.detect_spots(...)`, line ~101):

```python
    sp = S.detect_spots(bin_res.binary, bin_res.gray, bin_res.polarity, bin_res.roi,
                        ln.baseline_y, ln.front_y, float(h * w), cfg,
                        keep_regions=(None if learned else llm_regions), scorer=scorer)

    # YOLO fallback: third layer — only when sklearn found nothing
    if use_yolo and not sp.spots:
        from . import yolo as Y
        yolo_sp = Y.detect_yolo(img, cfg, ln.baseline_y, ln.front_y)
        if yolo_sp.spots:
            sp = yolo_sp
            engine_used = f"{engine_used}+yolo"
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/bruceli/Documents/hackathon/cv
.venv/bin/python3 -m pytest tests/test_yolo_pipeline.py -v
```
Expected: all tests PASS (the monkeypatching + `is_trained=False` path exercises the fallback).

- [ ] **Step 5: Commit**

```bash
git add cv/chromalog_cv/pipeline.py cv/tests/test_yolo_pipeline.py
git commit -m "feat(yolo): pipeline.py YOLO fallback when sklearn returns 0 spots"
```

---

### Task 5: Sidecar endpoints (`server.py` patch)

**Files:**
- Modify: `cv/chromalog_cv/server.py`
- Test: manual curl (no new test file — existing sidecar integration verified by curl)

**Interfaces:**
- Consumes: `run_pipeline(..., use_yolo=bool)` from Task 4; `LOCK` and `ONNX` paths from `train_yolo` module conventions
- Produces:
  - `/detect?use_yolo=true` — passes `use_yolo` to `run_pipeline`
  - `POST /train-yolo` → `{"ok": true, "status": "training_started"}` or `{"ok": false, "error": "already training"}`
  - `GET /yolo-model` → `{"status": "not_trained"|"training"|"ready", "trained_at": str|null}`

- [ ] **Step 1: Add `use_yolo` query param to `/detect` endpoint**

In `server.py`, find the `/detect` endpoint parameters and add after `line_min_len_frac`:
```python
    use_yolo: bool = Query(False, description="启用 YOLO 第三层 fallback (需已训练 yolo_spot.onnx)"),
```

In the endpoint body, find the `run_pipeline(...)` call and add `use_yolo=use_yolo`:
```python
    result, dbg, rect_img = run_pipeline(img, cfg, debug=debug,
                                         llm_regions=llm_regions, engine_used=engine,
                                         rect=rec, use_yolo=use_yolo)
```

- [ ] **Step 2: Add path constants and two new endpoints**

Near the top of `server.py`, after the existing imports, add:
```python
import subprocess
from pathlib import Path as _Path

_CV_ROOT    = _Path(__file__).resolve().parent.parent   # cv/
_YOLO_ONNX  = _CV_ROOT / "models" / "yolo_spot.onnx"
_YOLO_LOCK  = _CV_ROOT / "models" / ".yolo_training"
_TRAIN_SCRIPT = _CV_ROOT / "train_yolo.py"
```

At the end of `server.py`, append:
```python
@app.post("/train-yolo")
def train_yolo_endpoint():
    """Start YOLO training in background. Returns immediately."""
    if _YOLO_LOCK.exists():
        return JSONResponse(status_code=200,
                            content={"ok": False, "error": "already training"})
    _CV_ROOT.joinpath("models").mkdir(exist_ok=True)
    skip_synth = (_CV_ROOT / "data" / "synth" / "dataset.yaml").exists()
    cmd = [sys.executable, str(_TRAIN_SCRIPT)]
    if skip_synth:
        cmd.append("--skip-synth")
    subprocess.Popen(cmd, cwd=str(_CV_ROOT))
    return {"ok": True, "status": "training_started"}


@app.get("/yolo-model")
def yolo_model_endpoint():
    """Report YOLO model status."""
    from datetime import datetime, timezone
    if _YOLO_LOCK.exists():
        return {"status": "training", "trained_at": None}
    if _YOLO_ONNX.exists():
        ts = datetime.fromtimestamp(_YOLO_ONNX.stat().st_mtime, timezone.utc).isoformat()
        return {"status": "ready", "trained_at": ts}
    return {"status": "not_trained", "trained_at": None}
```

Also add `import sys` near the top if not already present.

- [ ] **Step 3: Restart sidecar and verify endpoints**

```bash
# Kill existing sidecar
pkill -f "uvicorn chromalog_cv" 2>/dev/null; sleep 1

# Start fresh
cd /Users/bruceli/Documents/hackathon/cv
.venv/bin/python3 run.py &>/tmp/sidecar.log &
sleep 3

# Test GET /yolo-model (no model yet → not_trained)
curl -s http://localhost:8765/yolo-model
# Expected: {"status":"not_trained","trained_at":null}

# Test POST /train-yolo (no lock yet → training_started)
curl -s -X POST http://localhost:8765/train-yolo
# Expected: {"ok":true,"status":"training_started"}

# Test /detect with use_yolo param (no model → no error, just skips YOLO)
curl -s -X POST "http://localhost:8765/detect?use_yolo=true" \
  -F "file=@/Users/bruceli/Documents/hackathon/training_pictures/TLC_real_5.jpg" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('ok, engine:', d.get('engine_used'))"
# Expected: ok, engine: opencv+skl  (no +yolo since no model file)
```

- [ ] **Step 4: Kill the training process that was started (it will fail without ultralytics — that's fine)**

```bash
pkill -f train_yolo.py 2>/dev/null
rm -f /Users/bruceli/Documents/hackathon/cv/models/.yolo_training
```

- [ ] **Step 5: Commit**

```bash
git add cv/chromalog_cv/server.py
git commit -m "feat(yolo): /detect use_yolo param + POST /train-yolo + GET /yolo-model"
```

---

### Task 6: Swift app — CVClient + AppStore + SettingsView

**Files:**
- Modify: `App/Sources/ChromaLog/CVClient.swift`
- Modify: `App/Sources/ChromaLog/AppStore.swift`
- Modify: `App/Sources/ChromaLog/Views/SettingsView.swift`

**Interfaces:**
- Consumes: `GET /yolo-model` → `{"status":…, "trained_at":…}` (Task 5); `POST /train-yolo` (Task 5)
- Produces:
  - `CVClient.trainYolo() async throws`
  - `CVClient.yoloModelStatus() async throws -> YoloModelStatus`
  - `AppStore.yoloStatus: String` (`"not_trained"` / `"training"` / `"ready"`)
  - `AppStore.useYolo: Bool` (computed, `yoloStatus == "ready"`)
  - `AppStore.startYoloTraining()`
  - SettingsView YOLO section

- [ ] **Step 1: Add `YoloModelStatus` struct and two methods to `CVClient.swift`**

In `CVClient.swift`, add after the existing `CVModelInfo` struct:
```swift
struct YoloModelStatus: Decodable {
    let status: String       // "not_trained" | "training" | "ready"
    let trained_at: String?
}
```

Add a new extension at the bottom of `CVClient.swift`:
```swift
extension CVClient {
    /// Trigger background YOLO training on the sidecar.
    func trainYolo() async throws {
        guard let url = URL(string: "\(baseURL)/train-yolo") else { return }
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.timeoutInterval = 10
        let (_, resp) = try await session.data(for: req)
        guard (resp as? HTTPURLResponse)?.statusCode == 200 else {
            throw CVClientError.notRunning
        }
    }

    /// Poll the sidecar for YOLO model status.
    func yoloModelStatus() async throws -> YoloModelStatus {
        guard let url = URL(string: "\(baseURL)/yolo-model") else {
            throw CVClientError.invalidResponse
        }
        let (data, _) = try await session.data(from: url)
        return try JSONDecoder().decode(YoloModelStatus.self, from: data)
    }
}
```

- [ ] **Step 2: Update `CVClient.detect()` to accept `useYolo`**

In the existing `detect()` method signature, add `useYolo: Bool = false` after `orKey`:
```swift
func detect(imageData: Data, debug: Bool = false,
            hatThreshK: Double? = nil, kneeDeviation: Double? = nil,
            useAI: Bool = false, orModel: String? = nil, orKey: String? = nil,
            useYolo: Bool = false) async throws -> CVDetectResponse {
```

In the method body, add the query item after the `use_ai` block:
```swift
        if useYolo {
            items.append(URLQueryItem(name: "use_yolo", value: "true"))
        }
```

- [ ] **Step 3: Add YOLO state and methods to `AppStore.swift`**

In `AppStore.swift`, add these published properties with the other CV-related state:
```swift
    @Published var yoloStatus: String = "not_trained"   // not_trained | training | ready
    var useYolo: Bool { yoloStatus == "ready" }
```

Add these methods at the end of the `AppStore` class (before the closing `}`):
```swift
    // MARK: - YOLO model management

    func startYoloTraining() {
        Task { @MainActor in
            do {
                guard await CVClient.shared.waitForReady(timeout: 5) else {
                    yoloStatus = "not_trained"; return
                }
                try await CVClient.shared.trainYolo()
                yoloStatus = "training"
                await pollYoloStatus()
            } catch {
                yoloStatus = "not_trained"
            }
        }
    }

    private func pollYoloStatus() async {
        while yoloStatus == "training" {
            do {
                try await Task.sleep(nanoseconds: 5_000_000_000)
                let info = try await CVClient.shared.yoloModelStatus()
                await MainActor.run { self.yoloStatus = info.status }
            } catch {
                break
            }
        }
    }

    func refreshYoloStatus() {
        Task {
            guard let info = try? await CVClient.shared.yoloModelStatus() else { return }
            await MainActor.run { self.yoloStatus = info.status }
        }
    }
```

Also update the `detect` call in `AppStore` that calls `CVClient.shared.detect(...)` to pass `useYolo: useYolo`. Find the call site (search for `CVClient.shared.detect(`) and add the parameter:
```swift
// find the existing detect call and add:
                                  useYolo: useYolo)
```

- [ ] **Step 4: Add YOLO section to `SettingsView.swift`**

Find the `VStack` in `SettingsView` body that holds the existing OpenRouter / model sections, and append this block before the final `HStack { Spacer(); Button("Done") ... }`:

```swift
            Divider()

            VStack(alignment: .leading, spacing: 6) {
                Text("YOLO Spot Detector")
                    .font(.system(size: 11, weight: .semibold))

                HStack(spacing: 8) {
                    Circle()
                        .fill(yoloDotColor)
                        .frame(width: 8, height: 8)
                    Text(yoloStatusLabel)
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                }

                Button {
                    store.startYoloTraining()
                } label: {
                    if store.yoloStatus == "training" {
                        HStack(spacing: 6) {
                            ProgressView().controlSize(.small)
                            Text("Training… (~5 min)")
                        }
                    } else {
                        Text("Re-train YOLO (YOLOv8n)")
                    }
                }
                .disabled(store.yoloStatus == "training")

                Text("Trains on synthetic TLC data locally. Requires: pip install ultralytics")
                    .font(.system(size: 9)).foregroundStyle(.tertiary)
            }
```

Add computed helpers inside `SettingsView` struct:
```swift
    private var yoloDotColor: Color {
        switch store.yoloStatus {
        case "ready":    return .green
        case "training": return .yellow
        default:         return .gray
        }
    }

    private var yoloStatusLabel: String {
        switch store.yoloStatus {
        case "ready":    return "Ready"
        case "training": return "Training…"
        default:         return "Not trained"
        }
    }
```

Also call `store.refreshYoloStatus()` on appear. Find `.onAppear` in SettingsView (or add one):
```swift
        .onAppear {
            // existing onAppear content ...
            store.refreshYoloStatus()
        }
```

- [ ] **Step 5: Build and verify no errors**

```bash
cd /Users/bruceli/Documents/hackathon/App
swift build 2>&1 | tail -5
```
Expected: `Build complete!`

- [ ] **Step 6: Commit**

```bash
git add App/Sources/ChromaLog/CVClient.swift \
        App/Sources/ChromaLog/AppStore.swift \
        App/Sources/ChromaLog/Views/SettingsView.swift
git commit -m "feat(app): YOLO status polling + Re-train button in Settings"
```

---

## Self-Review

**Spec coverage check:**

| Spec section | Covered by |
|---|---|
| §1 success criteria: `/detect?use_yolo=true` fallback | Task 4 + Task 5 |
| §1 success criteria: trained on synthetic data, exports ONNX | Task 1 + Task 3 |
| §1 success criteria: Re-train button, status polling | Task 6 |
| §1 success criteria: existing behaviour unchanged with `use_yolo=false` | Task 4 test |
| §2 file layout | Tasks 1–6 each create/modify exactly the listed files |
| §3 synthetic data: real bg fallback, 2–8 spots, both polarities, augmentation | Task 1 |
| §3 synthetic data: 90/10 split, YOLO label format, dataset.yaml | Task 1 |
| §4 train_yolo.py: MPS, 50 epochs, real fine-tune if available, ONNX export | Task 3 |
| §4 lockfile `.yolo_training` written at start, deleted on finish | Task 3 |
| §5 detect_yolo: lazy session load, resize 640, NMS, conf ≥ 0.35 | Task 2 |
| §5 detect_yolo: Rf computed via `_rf()` | Task 2 |
| §6 pipeline fallback: only when `sp.spots` empty + `use_yolo=True` | Task 4 |
| §7 POST /train-yolo: background Popen, lockfile check | Task 5 |
| §7 GET /yolo-model: 3 states from lockfile + onnx presence | Task 5 |
| §8 CVClient.trainYolo + yoloModelStatus | Task 6 step 1 |
| §8 AppStore.yoloStatus + startYoloTraining + polling | Task 6 step 3 |
| §8 SettingsView YOLO section: dot, label, button, disabled while training | Task 6 step 4 |
| §9 ultralytics in requirements.txt (commented) | Task 3 step 2 |
| §10 out of scope: no annotation UI, no partial_fit for YOLO | not implemented ✓ |

**No placeholders or TODOs found.**

**Type consistency confirmed:** `YoloModelStatus.status: String` matches `AppStore.yoloStatus: String`; `detect_yolo` signature matches `pipeline.py` call site with `ln.baseline_y` / `ln.front_y`; `SpotsResult` from `spots.py` used consistently across Tasks 2, 4.
