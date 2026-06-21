# YOLO Spot Detection — Design Spec (M5+ c)

**Date:** 2026-06-21
**Branch:** feat/yolo-spot-detection (to be created)
**Spec version:** 1.0

---

## 1. Context & Goal

ChromaLog's current pipeline (M5) detects TLC spots via:
OpenCV connected-components → sklearn SGD patch classifier → Rf output.

This spec adds **YOLOv8n** as a third-layer fallback (spec appendix D.3, model 3), activated only when the upstream layers return zero spots. YOLO is accessed via ONNX Runtime so the model is fully cross-platform.

**Success criteria:**
- `POST /detect?use_yolo=true` falls back to YOLO when sklearn finds 0 spots.
- YOLOv8n trained on synthetic data runs locally on Mac (MPS), exports to `models/yolo_spot.onnx`.
- User can re-trigger training from Settings; status is polled until ready.
- All existing behaviour with `use_yolo=false` (default) is unchanged.

---

## 2. File Layout

```
cv/
├── chromalog_cv/
│   ├── yolo.py          ← NEW: synthetic data gen + ONNX inference
│   ├── pipeline.py      ← PATCH: ~10 lines fallback at end of pipeline
│   └── server.py        ← PATCH: POST /train-yolo + GET /yolo-model
├── train_yolo.py        ← NEW: standalone training script (CLI, not sidecar)
models/
└── yolo_spot.onnx       ← training artifact (.gitignore already covers *.onnx)
cv/data/synth/           ← generated dataset (.gitignore, not committed)
```

**`yolo.py` public API (two functions only):**

```python
def generate_synthetic_dataset(real_images_dir, out_dir, n=2000) -> None
def detect_yolo(img: np.ndarray, cfg: Config,
                baseline_y: float | None = None,
                front_y: float | None = None) -> SpotsResult
```

`ultralytics` is imported with `try/except ImportError`; if absent, `detect_yolo` returns empty `SpotsResult` and sidecar continues normally.

---

## 3. Synthetic Data Generation

`generate_synthetic_dataset()` in `yolo.py`:

1. **Background**: randomly crop a blank region from real images in `training_pictures/`; fall back to solid colour + Gaussian noise if no real images available.
2. **Spots**: 2–8 Gaussian blobs per plate; random radius 4–18 px, random intensity, random x, y constrained between baseline and solvent front (both randomised per plate). Both polarity modes: dark-on-light (UV254) and light-on-dark (UV365).
3. **Augmentation**: brightness/contrast jitter, Gaussian noise, mild affine (±5° rotation, ±3% scale). **No large vertical stretch** — preserves Rf geometry (spec D.4).
4. **Labels**: bbox is known at generation time → written as YOLO `.txt` (`0 cx cy w h` normalised). Label quality is 100%.
5. **Split**: 1800 train / 200 val, written under `cv/data/synth/images/` and `cv/data/synth/labels/`.
6. **`dataset.yaml`**: written to `cv/data/synth/dataset.yaml` (ultralytics format, 1 class: `spot`).

---

## 4. Training Script `train_yolo.py`

```
cd cv
python train_yolo.py             # generate synth + train + export
python train_yolo.py --skip-synth  # skip generation (data already exists)
```

Steps:
1. Call `yolo.generate_synthetic_dataset()` (skipped with `--skip-synth`).
2. `YOLO("yolov8n.pt").train(data="data/synth/dataset.yaml", epochs=50, imgsz=640, device="mps")`. ~3–5 min on Apple Silicon.
3. If `cv/data/real/` exists (user-annotated real images in YOLO format), run a second fine-tune pass: `epochs=20`, lower `lr0`.
4. `model.export(format="onnx", opset=12, simplify=True)` → `models/yolo_spot.onnx`.
5. Smoke-test with ONNX Runtime: one inference pass, print latency.

**Training state files** (used by sidecar for status polling):
- `models/.yolo_training` — lockfile written at start, deleted on finish.
- `models/yolo_spot.onnx` — presence + mtime signals completion.

---

## 5. ONNX Inference in `yolo.py`

`detect_yolo(img, cfg, baseline_y, front_y) → SpotsResult`:

1. Lazily load `onnxruntime.InferenceSession("models/yolo_spot.onnx")` on first call; cache globally.
2. Pre-process: resize to 640×640, normalise to [0,1], shape `[1,3,640,640]`.
3. Post-process: parse YOLOv8 output tensor, apply NMS, filter by confidence ≥ 0.35, map bbox centres back to original image coordinates.
4. Compute Rf for each detection using existing `_rf(spot_y, baseline_y, front_y)`.
5. Return `SpotsResult` with same fields as OpenCV path — pipeline.py is unaware of the source.

---

## 6. Pipeline Integration (`pipeline.py`)

Addition at the end of `run_pipeline()`, after the sklearn scorer stage:

```python
if use_yolo and not result.spots:
    from . import yolo as Y
    yolo_result = Y.detect_yolo(img, cfg, ln.baseline_y, ln.front_y)
    if yolo_result.spots:
        result = yolo_result
        engine_used += "+yolo"
```

`use_yolo` is passed in from the `/detect` query param (default `False`). YOLO is called **only when sklearn returns 0 spots** — normal images never pay the YOLO inference cost.

---

## 7. Sidecar Endpoints (`server.py`)

### `POST /train-yolo`
- Checks no training already in progress (lockfile absent).
- Launches `python train_yolo.py --skip-synth` in background via `subprocess.Popen` if `data/synth/` already exists, else full run.
- Returns immediately: `{"ok": true, "status": "training_started"}`.
- Returns `{"ok": false, "error": "already training"}` if lockfile present.

### `GET /yolo-model`
```json
{
  "status": "not_trained" | "training" | "ready",
  "trained_at": "<ISO timestamp or null>"
}
```
Status logic:
- lockfile present → `"training"`
- `models/yolo_spot.onnx` exists + lockfile absent → `"ready"`
- neither → `"not_trained"`

---

## 8. Swift App Integration

### `CVClient.swift`
```swift
func trainYolo() async throws
func yoloModelStatus() async throws -> YoloModelStatus  // not_trained | training | ready
```

### `AppStore.swift`
- `@Published var yoloStatus: String = "not_trained"`
- `func startYoloTraining()`: calls `CVClient.trainYolo()`, then polls `yoloModelStatus()` every 5 s until `ready` or error.
- `var useYolo: Bool { yoloStatus == "ready" }` — automatically enables YOLO in `/detect` calls once model is present.

### `SettingsView.swift` — new YOLO section (below OpenRouter block)

```
┌─ YOLO Spot Detector ──────────────────────────────┐
│  Status: ● Ready  (trained 2026-06-21)            │
│                                                   │
│  [ Re-train YOLO (YOLOv8n) ]                      │
│  Trains on synthetic data. Takes ~5 min.          │
└───────────────────────────────────────────────────┘
```

- Button disabled + spinner while `yoloStatus == "training"`.
- Status dot: grey = not_trained, yellow = training, green = ready.
- On completion, `yoloStatus` updates to `"ready"` and all subsequent `/detect` calls include `use_yolo=true`.

---

## 9. Dependencies

Add to `requirements.txt` (commented, manual install):
```
#   ultralytics>=8.2   # YOLO training (run: pip install ultralytics)
#   onnxruntime>=1.17  # YOLO inference (already listed)
```

`onnxruntime` is already in requirements. `ultralytics` only needed for training (`train_yolo.py`); `yolo.py` inference path only needs `onnxruntime`.

---

## 10. Out of Scope

- Real-data annotation UI (user adds to `cv/data/real/` manually or via LabelImg).
- Active learning / automatic fine-tune from user corrections (YOLO doesn't support `partial_fit`; re-training is the only path).
- YOLO replacing sklearn (it's a fallback only).
- YOLOv8s or larger models.
