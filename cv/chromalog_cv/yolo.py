"""YOLO spot detector (M5+ c): synthetic dataset generation + ONNX inference.

ultralytics is only needed for training (train_yolo.py); this file uses only
onnxruntime at inference time and is importable with neither installed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

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
    real_images_dir: str | Path,
    out_dir: str | Path,
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
