"""斑点 patch 在线增量分类 (spec 附录 D.3 模型 2 / 设计 §3)。

OpenCV 提出候选, 本模块判每个候选"是不是真斑点", 用户每次手动矫正 -> partial_fit,
立即"越标越准"。patch = 质心为中心的固定窗口 (Swift Spot 只有质心, 无 bbox)。
模型 pickle 落盘于 cv/models/; YOLO 将来可整体热替换"提议器+打分器", 下游不变。
"""
from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from .config import Config

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
CLF_PATH = MODEL_DIR / "spot_clf.pkl"
SAMPLES_PATH = MODEL_DIR / "spot_samples.npz"


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def patch_features(img: np.ndarray, cx: float, cy: float, cfg: Config) -> np.ndarray:
    """質心為中心裁固定窗口 -> 定長特徵 (展平灰度 patch + [均值,標準差,Sobel能量])。"""
    gray = _to_gray(img)
    h, w = gray.shape[:2]
    win = max(8, int(cfg.clf_patch_frac * min(h, w)))
    half = win // 2
    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    x0 = max(0, min(x0, w - win)) if w >= win else 0
    y0 = max(0, min(y0, h - win)) if h >= win else 0
    crop = gray[y0:y0 + win, x0:x0 + win]
    if crop.size == 0:
        crop = gray
    patch = cv2.resize(crop, (cfg.clf_patch_size, cfg.clf_patch_size)).astype(np.float32) / 255.0
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1)
    stats = np.array([patch.mean(), patch.std(), float(np.sqrt(gx * gx + gy * gy).mean())],
                     dtype=np.float32)
    return np.concatenate([patch.ravel(), stats]).astype(np.float32)


class SpotClassifier:
    """在线增量二分类: 0=非斑点, 1=真斑点。StandardScaler + SGD(log_loss) 均用 partial_fit。"""

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.clf = SGDClassifier(loss="log_loss", random_state=0)
        self.n_samples = 0
        self._classes = np.array([0, 1])

    @property
    def is_trained(self) -> bool:
        return self.n_samples > 0

    def update(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, np.float32); y = np.asarray(y).astype(int)
        if X.shape[0] == 0:
            return
        self.scaler.partial_fit(X)
        Xs = self.scaler.transform(X)
        self.clf.partial_fit(Xs, y, classes=self._classes)
        self.n_samples += int(X.shape[0])

    def proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, np.float32)
        if not self.is_trained:
            return np.full(X.shape[0], 0.5, np.float32)
        Xs = self.scaler.transform(X)
        idx = list(self.clf.classes_).index(1)
        return self.clf.predict_proba(Xs)[:, idx].astype(np.float32)

    def save(self, path: Path = CLF_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path = CLF_PATH) -> "SpotClassifier":
        path = Path(path)
        if not path.exists():
            return cls()
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
            return obj if isinstance(obj, cls) else cls()
        except Exception:
            return cls()
