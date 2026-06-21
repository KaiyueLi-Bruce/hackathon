"""斑点 patch 在线增量分类 (spec 附录 D.3 模型 2 / 设计 §3)。

OpenCV 提出候选, 本模块判每个候选"是不是真斑点", 用户每次手动矫正 -> partial_fit,
立即"越标越准"。patch = 质心为中心的固定窗口 (Swift Spot 只有质心, 无 bbox)。
模型 pickle 落盘于 cv/models/; YOLO 将来可整体热替换"提议器+打分器", 下游不变。
"""
from __future__ import annotations

import math
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from .config import Config

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
CLF_PATH = MODEL_DIR / "spot_clf.pkl"
SAMPLES_PATH = MODEL_DIR / "spot_samples.npz"
LINE_CLF_PATH = MODEL_DIR / "line_clf.pkl"      # 基线/前沿 行分类器 (与 SpotClassifier 同构)
LINE_SAMPLES_PATH = MODEL_DIR / "line_samples.npz"


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


def row_features(img: np.ndarray, y_norm: float, cfg: Config) -> np.ndarray:
    """某一归一化行 y 的'是不是基线/前沿线'特征 (定长 6 维)。
    基线/前沿是细、暗、横贯板宽的线 -> 用 暗度/上下对比/横向均匀度/暗列占比/垂直边缘/位置。"""
    gray = _to_gray(img).astype(np.float32)
    h, w = gray.shape[:2]
    yc = int(round(y_norm * h))
    half = max(1, int(0.006 * h))
    def band(a, b):
        a = max(0, a); b = min(h, b)
        if b <= a:
            return np.full(w, 255.0, np.float32)
        return gray[a:b].mean(axis=0)
    center = band(yc - half, yc + half + 1)               # 该行(逐列均值)
    above = band(yc - 4 * half, yc - half)
    below = band(yc + half + 1, yc + 4 * half + 1)
    cm = float(center.mean())
    darkness = (255.0 - cm) / 255.0
    surround = ((above.mean() + below.mean()) / 2.0 - cm) / 255.0     # 比上下暗多少
    uniform = 1.0 - min(1.0, float(center.std()) / (cm + 1.0))        # 横向越均匀越像线
    thr = float(np.median(gray)) - 12.0
    dark_frac = float((center < thr).mean())                         # 暗列占比(横贯)
    vedge = abs(above.mean() - below.mean()) / 255.0
    return np.array([darkness, surround, uniform, dark_frac, vedge, float(y_norm)], np.float32)


def derive_line_samples(img, baseline_y, front_y, cfg: Config, rng_seed: int = 1):
    """从基线/前沿矫正派生 (X,y,counts): 两条线=正, 远离的随机行=负。"""
    feats, labels = [], []
    pos_ys = [v for v in (baseline_y, front_y) if v is not None]
    for v in pos_ys:
        feats.append(row_features(img, float(v), cfg)); labels.append(1)
    rng = np.random.RandomState(rng_seed)
    n_neg = max(4, 3 * len(pos_ys))
    added, tries = 0, 0
    while added < n_neg and tries < n_neg * 40 + 40:
        tries += 1
        ry = float(rng.uniform(0.05, 0.95))
        if all(abs(ry - v) > 0.06 for v in pos_ys):
            feats.append(row_features(img, ry, cfg)); labels.append(0); added += 1
    X = np.array(feats, np.float32) if feats else np.zeros((0, 6), np.float32)
    y = np.array(labels, int)
    return X, y, {"pos": int((y == 1).sum()), "neg": int((y == 0).sum())}


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


def _near(p, pts, tol):
    for q in pts:
        if math.hypot(p[0] - q[0], p[1] - q[1]) <= tol:
            return True
    return False


def derive_samples(img, final_pts, auto_pts, cfg: Config, rng_seed: int = 0):
    """从一次矫正派生 (X, y, counts)。坐标为归一化质心。"""
    h, w = img.shape[:2]
    tol = cfg.clf_match_frac * math.sqrt(2.0)          # 归一化对角线比例
    win_norm = cfg.clf_patch_frac                      # 易负与正样本至少隔一个窗口

    feats, labels = [], []

    def add(px, py, lab):
        feats.append(patch_features(img, px * w, py * h, cfg))
        labels.append(lab)

    pos = 0
    # 自动候选: 命中最终斑点=正, 否则硬负
    for c in auto_pts:
        if _near(c, final_pts, tol):
            add(c[0], c[1], 1); pos += 1
        else:
            add(c[0], c[1], 0)
    # 最终斑点里匹配不到任何候选的 = 用户新增, 正
    for s in final_pts:
        if not _near(s, auto_pts, tol):
            add(s[0], s[1], 1); pos += 1

    # 易负: 随机背景, 远离所有最终斑点
    rng = np.random.RandomState(rng_seed)
    n_easy = math.ceil(cfg.clf_easy_neg_ratio * pos)
    tries = 0
    added_easy = 0
    while added_easy < n_easy and tries < n_easy * 50 + 50:
        tries += 1
        px, py = float(rng.rand()), float(rng.rand())
        if not _near((px, py), final_pts, win_norm):
            add(px, py, 0); added_easy += 1

    X = np.array(feats, np.float32) if feats else np.zeros((0, cfg.clf_patch_size ** 2 + 3), np.float32)
    y = np.array(labels, int)
    counts = {"pos": int((y == 1).sum()), "neg": int((y == 0).sum())}
    return X, y, counts


def append_samples(X, y, path: Path = SAMPLES_PATH) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        d = np.load(path)
        X = np.vstack([d["X"], X]) if X.shape[0] else d["X"]
        y = np.concatenate([d["y"], y]) if y.shape[0] else d["y"]
    np.savez(path, X=X.astype(np.float32), y=y.astype(int))
    return int(X.shape[0])


def score_rows(img, y_norms, cfg: Config, clf: "SpotClassifier") -> np.ndarray:
    """对一批候选行打 P(是线) 分。clf 未训练时返回全 0.5。"""
    if not y_norms:
        return np.zeros(0, np.float32)
    X = np.array([row_features(img, float(v), cfg) for v in y_norms], np.float32)
    return clf.proba(X)


def model_info(clf_path: Path = CLF_PATH, samples_path: Path = SAMPLES_PATH) -> dict:
    def _count(p):
        p = Path(p)
        return int(np.load(p)["y"].shape[0]) if p.exists() else 0
    info = {"trained": Path(clf_path).exists(), "n_samples": _count(samples_path),
            "updated_at": None,
            "lines_trained": LINE_CLF_PATH.exists(), "line_samples": _count(LINE_SAMPLES_PATH)}
    if Path(clf_path).exists():
        info["updated_at"] = datetime.fromtimestamp(os.path.getmtime(clf_path), timezone.utc).isoformat()
    return info


def apply_correction(img, final_pts, auto_pts, cfg: Config,
                     baseline_y=None, front_y=None) -> dict:
    # 斑点分类器
    X, y, counts = derive_samples(img, final_pts, auto_pts, cfg)
    clf = SpotClassifier.load(CLF_PATH)
    clf.update(X, y)
    clf.save(CLF_PATH)
    total = append_samples(X, y, SAMPLES_PATH)
    out = {"ok": True, "batch": counts, "trained_total": total}

    # 线分类器 (基线/前沿)
    if baseline_y is not None or front_y is not None:
        lX, ly, lcounts = derive_line_samples(img, baseline_y, front_y, cfg)
        if lX.shape[0]:
            lclf = SpotClassifier.load(LINE_CLF_PATH)
            lclf.update(lX, ly)
            lclf.save(LINE_CLF_PATH)
            ltotal = append_samples(lX, ly, LINE_SAMPLES_PATH)
            out["line_batch"] = lcounts
            out["line_trained_total"] = ltotal
    return out
