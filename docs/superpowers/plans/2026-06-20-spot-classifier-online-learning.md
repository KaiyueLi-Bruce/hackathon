# Spot-Classifier Online Learning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the CV engine learn from manual spot corrections (sklearn `SGDClassifier.partial_fit`) so the next detection on similar plates suppresses false positives (plate corners, glare, background).

**Architecture:** OpenCV stays the candidate *proposer*; a sklearn patch classifier becomes the *scorer* that filters candidates. On save, the app posts the rectified image + final spots + original auto-candidates to a new `/learn` endpoint; the sidecar derives positive/hard-negative/easy-negative patches and does an online `partial_fit`. A trained model, if present, replaces the area-knee filter at detect time. YOLO is left as a documented offline-swap seam — not wired this round.

**Tech Stack:** Python 3.14 sidecar (FastAPI, OpenCV, NumPy, scikit-learn, pickle), SwiftUI app (SwiftPM, URLSession), pytest.

## Global Constraints

- Engine is platform-independent Python; CPU-only; no GPU deps. (spec §5, D.3)
- **This round installs `scikit-learn` only.** Do NOT add `onnxruntime`, `skl2onnx`, `ultralytics`, or `torch`. (design §7)
- Model + samples persist to `cv/models/` and are **gitignored** (grow with user data). (design §6)
- Patches are **centroid + fixed window** (the Swift `Spot` has only a normalized centroid `point`, no bbox). Training and inference use the same extractor. (design §3.1, refined)
- The rectified image used for `/learn` MUST be the exact image the app displays (`plateImage`, set from `/detect`'s `image_b64`) so coordinates align. (design §9)
- Model is authoritative when trained: if the scorer rejects all candidates, the result is empty (no area-knee fallback) — the model is trusted to have learned rejections. (design §4)
- All coordinates normalized to the rectified image, origin top-left, y down. (existing convention)
- Detection behavior with no trained model MUST be byte-for-byte the current behavior (AI-region filter / area-knee). (design §8 regression)

---

## File Structure

**Python (sidecar):**
- Create `cv/chromalog_cv/learn.py` — patch features, `SpotClassifier`, sample derivation, model paths/IO.
- Create `cv/tests/__init__.py`, `cv/tests/test_learn.py` — unit + e2e tests.
- Modify `cv/chromalog_cv/config.py` — add classifier knobs.
- Modify `cv/chromalog_cv/spots.py` — `detect_spots` gains `scorer=` param + precedence.
- Modify `cv/chromalog_cv/pipeline.py` — load classifier, build scorer, set `learned` + `engine_used`.
- Modify `cv/chromalog_cv/server.py` — `POST /learn`, `GET /model`.
- Modify `cv/requirements.txt` — uncomment/add scikit-learn.
- Modify `.gitignore` — add `cv/models/`.

**Swift (app):**
- Modify `App/Sources/ChromaLog/CVClient.swift` — `learn(...)`, `modelInfo()`, response structs.
- Modify `App/Sources/ChromaLog/AppStore.swift` — `autoCandidates` snapshot, call `learn` after save, `modelTrainedCount`.
- Modify `App/Sources/ChromaLog/Views/InspectorView.swift` — "Learned from N corrections" status line.

---

## Task 1: Dependencies, config knobs, gitignore

**Files:**
- Modify: `cv/requirements.txt`
- Modify: `cv/chromalog_cv/config.py:9-12`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `Config` fields `clf_patch_frac: float`, `clf_patch_size: int`, `spot_clf_thresh: float`, `clf_match_frac: float`, `clf_easy_neg_ratio: float`.

- [ ] **Step 1: Add scikit-learn + pytest to the venv**

Run:
```bash
cd /Users/bruceli/Documents/hackathon/cv && .venv/bin/pip install "scikit-learn>=1.4" pytest
```
Expected: installs successfully; `.venv/bin/python -c "import sklearn, pytest"` exits 0.

- [ ] **Step 2: Record scikit-learn in requirements.txt**

In `cv/requirements.txt`, change the commented upgrade line so scikit-learn is a real dependency. Replace:
```
# 升级项 (M5+, 现在不强制安装):
#   onnxruntime>=1.17   # 跨平台推理 (板分割 U-Net / sklearn / YOLO)
#   scikit-learn        # 斑点 patch 增量分类
#   skl2onnx            # sklearn -> ONNX
```
with:
```
# 斑点 patch 在线增量分类 (M5+):
scikit-learn>=1.4
# 升级项 (未启用):
#   onnxruntime>=1.17   # 跨平台推理 (板分割 U-Net / YOLO / sklearn->onnx)
#   skl2onnx            # sklearn -> ONNX 导出
#   ultralytics torch   # YOLO 训练/推理
```

- [ ] **Step 3: Add classifier knobs to Config**

In `cv/chromalog_cv/config.py`, after line `rectify_cv_trust: float = 0.45 ...` (end of the `① 正畸` block), add:
```python
    # ---- 斑点 patch 在线增量分类 (learn.py) ----
    clf_patch_frac: float = 0.10    # patch 边长相对板短边 (质心为中心的固定窗口)
    clf_patch_size: int = 24        # patch 统一缩放到该方形像素后提特征
    spot_clf_thresh: float = 0.5    # P(real) >= 此值保留 (已训练时替代面积拐点)
    clf_match_frac: float = 0.04    # 候选↔最终斑点 质心距 < 该比例(对角线) 视为同一个
    clf_easy_neg_ratio: float = 1.0 # 易负(随机背景)样本数 / 正样本数
```

- [ ] **Step 4: Gitignore the models dir**

In `.gitignore`, under the `# ---- CV debug / test artifacts ----` block, add:
```
# ---- 学习模型 + 训练样本 (随用户数据增长) ----
cv/models/
```

- [ ] **Step 5: Verify config loads**

Run:
```bash
cd /Users/bruceli/Documents/hackathon/cv && .venv/bin/python -c "from chromalog_cv.config import Config; c=Config(); print(c.clf_patch_frac, c.clf_patch_size, c.spot_clf_thresh, c.clf_match_frac, c.clf_easy_neg_ratio)"
```
Expected: `0.1 24 0.5 0.04 1.0`

- [ ] **Step 6: Commit**

```bash
cd /Users/bruceli/Documents/hackathon
git add cv/requirements.txt cv/chromalog_cv/config.py .gitignore
git commit -m "feat(learn): add scikit-learn dep, classifier config knobs, gitignore models/"
```

---

## Task 2: Patch feature extractor

**Files:**
- Create: `cv/chromalog_cv/learn.py`
- Create: `cv/tests/__init__.py`
- Create: `cv/tests/test_learn.py`

**Interfaces:**
- Produces: `patch_features(img: np.ndarray, cx: float, cy: float, cfg: Config) -> np.ndarray`
  — `img` is BGR or gray (rectified plate); `cx,cy` are **pixel** centroid; returns a fixed-length 1-D `float32` vector of length `cfg.clf_patch_size**2 + 3` (flattened normalized grayscale patch + `[mean, std, sobel_energy]`). Window side = `int(cfg.clf_patch_frac * min(h,w))`, clamped ≥ 8; crop is clamped to image bounds then resized to `clf_patch_size`.

- [ ] **Step 1: Write the failing test**

Create `cv/tests/__init__.py` (empty file).

Create `cv/tests/test_learn.py`:
```python
import numpy as np
from chromalog_cv.config import Config
from chromalog_cv import learn


def test_patch_features_fixed_length_and_scale_invariant():
    cfg = Config()
    expected_len = cfg.clf_patch_size ** 2 + 3
    big = np.full((400, 300, 3), 200, np.uint8)
    small = np.full((120, 90, 3), 200, np.uint8)
    fb = learn.patch_features(big, 150, 200, cfg)
    fs = learn.patch_features(small, 45, 60, cfg)
    assert fb.shape == (expected_len,)
    assert fs.shape == (expected_len,)        # length independent of image size
    assert fb.dtype == np.float32


def test_patch_features_centroid_near_edge_does_not_crash():
    cfg = Config()
    img = np.zeros((100, 100), np.uint8)
    f = learn.patch_features(img, 1, 1, cfg)   # gray input, corner centroid
    assert f.shape == (cfg.clf_patch_size ** 2 + 3,)
    assert np.all(np.isfinite(f))
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /Users/bruceli/Documents/hackathon/cv && .venv/bin/python -m pytest tests/test_learn.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'chromalog_cv.learn'`.

- [ ] **Step 3: Write minimal implementation**

Create `cv/chromalog_cv/learn.py`:
```python
"""斑点 patch 在线增量分类 (spec 附录 D.3 模型 2 / 设计 §3)。

OpenCV 提出候选, 本模块判每个候选"是不是真斑点", 用户每次手动矫正 -> partial_fit,
立即"越标越准"。patch = 质心为中心的固定窗口 (Swift Spot 只有质心, 无 bbox)。
模型 pickle 落盘于 cv/models/; YOLO 将来可整体热替换"提议器+打分器", 下游不变。
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import Config

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
CLF_PATH = MODEL_DIR / "spot_clf.pkl"
SAMPLES_PATH = MODEL_DIR / "spot_samples.npz"


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def patch_features(img: np.ndarray, cx: float, cy: float, cfg: Config) -> np.ndarray:
    """质心为中心裁固定窗口 -> 定长特征 (展平灰度 patch + [均值,标准差,Sobel能量])。"""
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
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cd /Users/bruceli/Documents/hackathon/cv && .venv/bin/python -m pytest tests/test_learn.py -v
```
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
cd /Users/bruceli/Documents/hackathon
git add cv/chromalog_cv/learn.py cv/tests/__init__.py cv/tests/test_learn.py
git commit -m "feat(learn): patch_features extractor + tests"
```

---

## Task 3: SpotClassifier (online partial_fit + persistence)

**Files:**
- Modify: `cv/chromalog_cv/learn.py`
- Modify: `cv/tests/test_learn.py`

**Interfaces:**
- Consumes: `patch_features` (Task 2).
- Produces:
  - `class SpotClassifier` with: `update(X: np.ndarray, y: np.ndarray) -> None` (online partial_fit), `proba(X: np.ndarray) -> np.ndarray` (P(real) per row), `is_trained: bool`, `n_samples: int`, `save(path=CLF_PATH) -> None`, classmethod `load(path=CLF_PATH) -> "SpotClassifier"` (returns an untrained instance if file missing).
  - `X` is a 2-D array `(n, feat_len)`, `y` is 1-D `{0,1}`.

- [ ] **Step 1: Write the failing test**

Append to `cv/tests/test_learn.py`:
```python
def test_classifier_learns_separable_data(tmp_path):
    rng = np.random.RandomState(0)
    pos = rng.normal(0.8, 0.05, size=(20, 5)).astype(np.float32)
    neg = rng.normal(0.2, 0.05, size=(20, 5)).astype(np.float32)
    X = np.vstack([pos, neg]); y = np.array([1] * 20 + [0] * 20)
    clf = learn.SpotClassifier()
    assert clf.is_trained is False
    clf.update(X, y)
    assert clf.is_trained is True
    assert clf.n_samples == 40
    p = clf.proba(np.array([[0.8, 0.8, 0.8, 0.8, 0.8],
                            [0.2, 0.2, 0.2, 0.2, 0.2]], np.float32))
    assert p[0] > 0.5 > p[1]


def test_classifier_save_load_roundtrip(tmp_path):
    path = tmp_path / "clf.pkl"
    X = np.random.RandomState(1).rand(10, 5).astype(np.float32)
    y = np.array([1, 0] * 5)
    clf = learn.SpotClassifier(); clf.update(X, y); clf.save(path)
    loaded = learn.SpotClassifier.load(path)
    assert loaded.is_trained and loaded.n_samples == 10
    assert np.allclose(loaded.proba(X), clf.proba(X))


def test_classifier_load_missing_returns_untrained(tmp_path):
    clf = learn.SpotClassifier.load(tmp_path / "nope.pkl")
    assert clf.is_trained is False and clf.n_samples == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /Users/bruceli/Documents/hackathon/cv && .venv/bin/python -m pytest tests/test_learn.py -k classifier -v
```
Expected: FAIL — `AttributeError: module 'chromalog_cv.learn' has no attribute 'SpotClassifier'`.

- [ ] **Step 3: Write minimal implementation**

Add to `cv/chromalog_cv/learn.py` (imports at top: add `import pickle`):
```python
import pickle

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler


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
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cd /Users/bruceli/Documents/hackathon/cv && .venv/bin/python -m pytest tests/test_learn.py -v
```
Expected: PASS (all tests, 5 passed).

- [ ] **Step 5: Commit**

```bash
cd /Users/bruceli/Documents/hackathon
git add cv/chromalog_cv/learn.py cv/tests/test_learn.py
git commit -m "feat(learn): SpotClassifier online partial_fit + pickle persistence"
```

---

## Task 4: Sample derivation from corrections

**Files:**
- Modify: `cv/chromalog_cv/learn.py`
- Modify: `cv/tests/test_learn.py`

**Interfaces:**
- Consumes: `patch_features` (Task 2), `Config`.
- Produces: `derive_samples(img, final_pts, auto_pts, cfg, rng_seed=0) -> tuple[np.ndarray, np.ndarray, dict]`
  - `img`: rectified BGR/gray. `final_pts`, `auto_pts`: lists of **normalized** `(x, y)` tuples.
  - Returns `(X, y, counts)` where `X` is `(n, feat_len) float32`, `y` is `(n,) int {0,1}`, `counts` is `{"pos": int, "neg": int}`.
  - Rules: auto candidate within `clf_match_frac` (× normalized diagonal = √2) of any final spot → **positive**; unmatched auto candidate → **hard negative**; final spot matching no candidate → **positive** (user-added); then sample `ceil(clf_easy_neg_ratio × pos)` random in-plate points ≥ one window away from every final spot → **easy negatives**.

- [ ] **Step 1: Write the failing test**

Append to `cv/tests/test_learn.py`:
```python
def test_derive_samples_splits_pos_hardneg_easyneg():
    cfg = Config()
    img = np.zeros((200, 200, 3), np.uint8)
    final_pts = [(0.50, 0.50)]                  # one real spot
    auto_pts = [(0.505, 0.495),                 # ~matches final -> positive
                (0.05, 0.05)]                   # corner, user deleted -> hard negative
    X, y, counts = learn.derive_samples(img, final_pts, auto_pts, cfg)
    # 1 matched candidate + 0 user-added = 1 positive; 1 hard neg; easy negs = ceil(1*1)=1
    assert counts["pos"] == 1
    assert counts["neg"] == 2                   # 1 hard + 1 easy
    assert X.shape[0] == 3 and X.shape[1] == cfg.clf_patch_size ** 2 + 3
    assert set(y.tolist()) == {0, 1}
    assert int((y == 1).sum()) == 1


def test_derive_samples_user_added_spot_is_positive():
    cfg = Config()
    img = np.zeros((200, 200, 3), np.uint8)
    final_pts = [(0.5, 0.5)]                     # present in final
    auto_pts = []                                # but never auto-detected -> user added
    X, y, counts = learn.derive_samples(img, final_pts, auto_pts, cfg)
    assert counts["pos"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /Users/bruceli/Documents/hackathon/cv && .venv/bin/python -m pytest tests/test_learn.py -k derive -v
```
Expected: FAIL — `AttributeError: ... has no attribute 'derive_samples'`.

- [ ] **Step 3: Write minimal implementation**

Add to `cv/chromalog_cv/learn.py` (add `import math` at top):
```python
import math


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
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cd /Users/bruceli/Documents/hackathon/cv && .venv/bin/python -m pytest tests/test_learn.py -v
```
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
cd /Users/bruceli/Documents/hackathon
git add cv/chromalog_cv/learn.py cv/tests/test_learn.py
git commit -m "feat(learn): derive_samples (pos / hard-neg / easy-neg) from corrections"
```

---

## Task 5: Persisted sample store + apply_correction wrapper

**Files:**
- Modify: `cv/chromalog_cv/learn.py`
- Modify: `cv/tests/test_learn.py`

**Interfaces:**
- Consumes: `derive_samples`, `SpotClassifier` (Tasks 3-4).
- Produces:
  - `append_samples(X, y, path=SAMPLES_PATH) -> int` — append to an `.npz` archive (`X`, `y`), returns total stored rows.
  - `model_info(clf_path=CLF_PATH, samples_path=SAMPLES_PATH) -> dict` — `{"trained": bool, "n_samples": int, "updated_at": str|None}` (`updated_at` = ISO mtime of clf file or None).
  - `apply_correction(img, final_pts, auto_pts, cfg) -> dict` — derive → `SpotClassifier.load().update(...)` → `save()` → `append_samples(...)`; returns `{"ok": True, "batch": counts, "trained_total": int}`.

- [ ] **Step 1: Write the failing test**

Append to `cv/tests/test_learn.py`:
```python
def test_apply_correction_trains_and_persists(tmp_path, monkeypatch):
    monkeypatch.setattr(learn, "CLF_PATH", tmp_path / "clf.pkl")
    monkeypatch.setattr(learn, "SAMPLES_PATH", tmp_path / "s.npz")
    cfg = Config()
    img = np.zeros((200, 200, 3), np.uint8)
    out = learn.apply_correction(img, [(0.5, 0.5)], [(0.5, 0.5), (0.05, 0.05)], cfg)
    assert out["ok"] is True
    assert out["batch"]["pos"] >= 1 and out["batch"]["neg"] >= 1
    info = learn.model_info(tmp_path / "clf.pkl", tmp_path / "s.npz")
    assert info["trained"] is True
    assert info["n_samples"] == out["trained_total"]
    assert info["updated_at"] is not None


def test_model_info_untrained(tmp_path):
    info = learn.model_info(tmp_path / "none.pkl", tmp_path / "none.npz")
    assert info == {"trained": False, "n_samples": 0, "updated_at": None}
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /Users/bruceli/Documents/hackathon/cv && .venv/bin/python -m pytest tests/test_learn.py -k "apply or model_info" -v
```
Expected: FAIL — `AttributeError: ... 'apply_correction'`.

- [ ] **Step 3: Write minimal implementation**

Add to `cv/chromalog_cv/learn.py` (add `from datetime import datetime, timezone` and `import os` at top):
```python
import os
from datetime import datetime, timezone


def append_samples(X, y, path: Path = SAMPLES_PATH) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        d = np.load(path)
        X = np.vstack([d["X"], X]) if X.shape[0] else d["X"]
        y = np.concatenate([d["y"], y]) if y.shape[0] else d["y"]
    np.savez(path, X=X.astype(np.float32), y=y.astype(int))
    return int(X.shape[0])


def model_info(clf_path: Path = CLF_PATH, samples_path: Path = SAMPLES_PATH) -> dict:
    clf_path, samples_path = Path(clf_path), Path(samples_path)
    if not clf_path.exists():
        return {"trained": False, "n_samples": 0, "updated_at": None}
    n = 0
    if samples_path.exists():
        n = int(np.load(samples_path)["y"].shape[0])
    ts = datetime.fromtimestamp(os.path.getmtime(clf_path), timezone.utc).isoformat()
    return {"trained": True, "n_samples": n, "updated_at": ts}


def apply_correction(img, final_pts, auto_pts, cfg: Config) -> dict:
    X, y, counts = derive_samples(img, final_pts, auto_pts, cfg)
    clf = SpotClassifier.load(CLF_PATH)
    clf.update(X, y)
    clf.save(CLF_PATH)
    total = append_samples(X, y, SAMPLES_PATH)
    return {"ok": True, "batch": counts, "trained_total": total}
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cd /Users/bruceli/Documents/hackathon/cv && .venv/bin/python -m pytest tests/test_learn.py -v
```
Expected: PASS (9 passed).

- [ ] **Step 5: Commit**

```bash
cd /Users/bruceli/Documents/hackathon
git add cv/chromalog_cv/learn.py cv/tests/test_learn.py
git commit -m "feat(learn): persisted sample store, model_info, apply_correction"
```

---

## Task 6: Wire scorer into detect_spots

**Files:**
- Modify: `cv/chromalog_cv/spots.py:137-187`
- Modify: `cv/tests/test_learn.py`

**Interfaces:**
- Consumes: nothing new (scorer is a plain callable).
- Produces: `detect_spots(..., keep_regions=None, scorer=None)` — `scorer: Callable[[Spot], float] | None`. When `scorer` is not None it is authoritative: keep spots with `scorer(s) >= cfg.spot_clf_thresh`, no fallback. When `scorer` is None, behavior is exactly the current keep_regions/area-knee precedence.

- [ ] **Step 1: Write the failing test**

Append to `cv/tests/test_learn.py`:
```python
from chromalog_cv import spots as S


def _spot(x, y):
    return S.Spot(x=x, y=y, bbox=(int(x) - 2, int(y) - 2, 4, 4), area=16, lane=0, rf=None)


def test_detect_spots_scorer_filters_below_threshold(monkeypatch):
    cfg = Config()
    binary = np.zeros((100, 100), np.uint8)
    # two blobs: one will score high, one low
    cv = __import__("cv2")
    cv.circle(binary, (30, 50), 3, 255, -1)
    cv.circle(binary, (70, 50), 3, 255, -1)
    gray = np.zeros((100, 100), np.uint8)
    roi = (0, 0, 100, 100)
    scorer = lambda s: 0.9 if s.x < 50 else 0.1   # keep left, drop right
    res = S.detect_spots(binary, gray, "dark_on_light", roi, None, None,
                         float(100 * 100), cfg, scorer=scorer)
    xs = sorted(round(s.x) for s in res.spots)
    assert all(x < 50 for x in xs)               # only the high-scored blob survives
    assert len(res.spots) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /Users/bruceli/Documents/hackathon/cv && .venv/bin/python -m pytest tests/test_learn.py -k scorer_filters -v
```
Expected: FAIL — `TypeError: detect_spots() got an unexpected keyword argument 'scorer'`.

- [ ] **Step 3: Write minimal implementation**

In `cv/chromalog_cv/spots.py`, change the signature (line ~137):
```python
def detect_spots(binary: np.ndarray, gray: np.ndarray, polarity: str, roi,
                 baseline_y, front_y, plate_area: float, cfg: Config,
                 keep_regions=None, scorer=None) -> SpotsResult:
```
Then replace the current filter block:
```python
    if keep_regions is not None:
        filtered = _filter_by_regions(spots, keep_regions, plate_w, plate_h, cfg)
        # 兜底(Plan C): ...
        spots = filtered if filtered else _area_knee_cut(spots, cfg)
    else:
        spots = _area_knee_cut(spots, cfg)
```
with:
```python
    if scorer is not None:
        # 已训练分类器: 打分过滤, 模型为准 (替代面积拐点, 无兜底 —— 设计 §4)
        spots = [s for s in spots if scorer(s) >= cfg.spot_clf_thresh]
    elif keep_regions is not None:
        filtered = _filter_by_regions(spots, keep_regions, plate_w, plate_h, cfg)
        # 兜底(Plan C): AI 粗框为空/坐标不可靠时与候选零重叠 -> 会清空所有候选。
        # 弱模型不该删掉好结果, 只该补短板: 此时回退纯 OpenCV 面积拐点压噪, 而非返回空。
        spots = filtered if filtered else _area_knee_cut(spots, cfg)
    else:
        spots = _area_knee_cut(spots, cfg)
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cd /Users/bruceli/Documents/hackathon/cv && .venv/bin/python -m pytest tests/test_learn.py -v
```
Expected: PASS (10 passed).

- [ ] **Step 5: Commit**

```bash
cd /Users/bruceli/Documents/hackathon
git add cv/chromalog_cv/spots.py cv/tests/test_learn.py
git commit -m "feat(learn): detect_spots gains authoritative scorer stage"
```

---

## Task 7: Load classifier in pipeline + report `learned`

**Files:**
- Modify: `cv/chromalog_cv/pipeline.py:29-49` (PipelineResult), `:51-115` (run_pipeline)
- Modify: `cv/tests/test_learn.py`

**Interfaces:**
- Consumes: `learn.SpotClassifier`, `learn.patch_features`, `detect_spots(scorer=)` (Tasks 2-3, 6).
- Produces: `PipelineResult.learned: bool` (default False). `run_pipeline` loads `SpotClassifier.load()`; if trained, builds `scorer = lambda s: clf.proba(patch_features(img, s.x, s.y, cfg)[None, :])[0]`, passes it to `detect_spots`, sets `result.learned=True` and appends `+skl` to `engine_used`. If untrained, `scorer=None` (current behavior, `learned=False`).

- [ ] **Step 1: Write the failing test**

Append to `cv/tests/test_learn.py`:
```python
from chromalog_cv.pipeline import run_pipeline


def test_pipeline_reports_learned_false_when_untrained(tmp_path, monkeypatch):
    monkeypatch.setattr(learn, "CLF_PATH", tmp_path / "absent.pkl")
    img = cv2_imread_real2()
    result, _, _ = run_pipeline(img)
    assert result.learned is False
    assert "skl" not in (result.engine_used or "")


def cv2_imread_real2():
    import cv2, os
    p = os.path.join(os.path.dirname(__file__), "..", "..", "training_pictures", "TLC_real_2.jpg")
    img = cv2.imread(p)
    assert img is not None, p
    return img
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /Users/bruceli/Documents/hackathon/cv && .venv/bin/python -m pytest tests/test_learn.py -k learned_false -v
```
Expected: FAIL — `AttributeError: 'PipelineResult' object has no attribute 'learned'`.

- [ ] **Step 3: Write minimal implementation**

In `cv/chromalog_cv/pipeline.py`:

(a) Add import near the others (`from . import enhance as E`):
```python
from . import learn as LN
```
(b) In `PipelineResult` add field after `engine_used`:
```python
    engine_used: str = "opencv"   # opencv | ai+opencv | yolo
    learned: bool = False         # 本次检测是否用了 sklearn 学习模型
```
(c) In `run_pipeline`, replace the spot-detection call block:
```python
    # ⑤ 斑点候选 (形状无关) + ⑥ 泳道; 有 AI 粗框则用其精修
    sp = S.detect_spots(bin_res.binary, bin_res.gray, bin_res.polarity, bin_res.roi,
                        ln.baseline_y, ln.front_y, float(h * w), cfg,
                        keep_regions=llm_regions)
```
with:
```python
    # ⑤ 斑点候选 (形状无关) + ⑥ 泳道。优先级: 学习模型 > AI 粗框 > 面积拐点。
    clf = LN.SpotClassifier.load()
    scorer = None
    learned = False
    if clf.is_trained:
        def scorer(s):  # noqa: E306 - 闭包打分器 (img=正畸BGR, 与候选同坐标系)
            f = LN.patch_features(img, s.x, s.y, cfg)
            return float(clf.proba(f[None, :])[0])
        learned = True
        engine_used = f"{engine_used}+skl"
    sp = S.detect_spots(bin_res.binary, bin_res.gray, bin_res.polarity, bin_res.roi,
                        ln.baseline_y, ln.front_y, float(h * w), cfg,
                        keep_regions=(None if learned else llm_regions), scorer=scorer)
```
(d) In the `result = PipelineResult(...)` constructor, add `learned=learned,` and ensure `engine_used=engine_used` (already passed).

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cd /Users/bruceli/Documents/hackathon/cv && .venv/bin/python -m pytest tests/test_learn.py -v
```
Expected: PASS (11 passed).

- [ ] **Step 5: Commit**

```bash
cd /Users/bruceli/Documents/hackathon
git add cv/chromalog_cv/pipeline.py cv/tests/test_learn.py
git commit -m "feat(learn): pipeline loads classifier, reports learned + engine_used+skl"
```

---

## Task 8: End-to-end learning test (suppress a false positive)

**Files:**
- Modify: `cv/tests/test_learn.py`

**Interfaces:**
- Consumes: `run_pipeline`, `apply_correction`, `learn.CLF_PATH/SAMPLES_PATH` (Tasks 5, 7).

- [ ] **Step 1: Write the failing test (behavior assertion)**

Append to `cv/tests/test_learn.py`:
```python
def test_learning_suppresses_taught_false_positive(tmp_path, monkeypatch):
    monkeypatch.setattr(learn, "CLF_PATH", tmp_path / "clf.pkl")
    monkeypatch.setattr(learn, "SAMPLES_PATH", tmp_path / "s.npz")
    from chromalog_cv.config import Config
    img = cv2_imread_real2()
    result0, _, _ = run_pipeline(img)
    assert len(result0.spots) >= 1
    assert result0.learned is False

    # Teach: keep the real spots, but mark each detected spot as a deleted
    # candidate that is ALSO a real spot for half, and teach background as neg.
    cfg = Config()
    # rectified image used by the pipeline = the displayed base; re-run rectify to get it
    from chromalog_cv import rectify as R
    rec = R.rectify(img, cfg)
    rimg = rec.image
    final_pts = [(s["x"], s["y"]) for s in result0.spots]
    # Teach the same spots as positives + background negatives over several rounds
    for _ in range(6):
        learn.apply_correction(rimg, final_pts, final_pts, cfg)

    result1, _, _ = run_pipeline(img)
    assert result1.learned is True
    assert "skl" in result1.engine_used
    # taught real spots should survive (model keeps positives it was trained on)
    assert len(result1.spots) >= 1
```

- [ ] **Step 2: Run test to verify it fails (or errors) before final wiring**

Run:
```bash
cd /Users/bruceli/Documents/hackathon/cv && .venv/bin/python -m pytest tests/test_learn.py -k suppresses -v
```
Expected: PASS if Tasks 5+7 are correct. If FAIL, debug the scorer coordinate alignment (img passed to scorer must be the rectified BGR — same as `R.rectify(img,cfg).image`).

- [ ] **Step 3: Run the full suite**

Run:
```bash
cd /Users/bruceli/Documents/hackathon/cv && .venv/bin/python -m pytest tests/ -v
```
Expected: PASS (12 passed).

- [ ] **Step 4: Manual sanity via CLI (no AI, with a trained model present)**

Run:
```bash
cd /Users/bruceli/Documents/hackathon/cv && .venv/bin/python run.py ../training_pictures/TLC_real_2.jpg 2>/dev/null | .venv/bin/python -c "import sys,json;d=json.load(sys.stdin);print('learned?', d.get('learned'), 'engine', d['engine_used'], 'spots', len(d['spots']))"
```
Expected: prints `learned? False engine opencv spots N` (no model trained in repo working dir; confirms graceful default).

- [ ] **Step 5: Commit**

```bash
cd /Users/bruceli/Documents/hackathon
git add cv/tests/test_learn.py
git commit -m "test(learn): e2e detect -> teach -> re-detect keeps taught spots"
```

---

## Task 9: Sidecar `/learn` and `/model` endpoints

**Files:**
- Modify: `cv/chromalog_cv/server.py`
- Modify: `cv/tests/test_learn.py`

**Interfaces:**
- Consumes: `learn.apply_correction`, `learn.model_info`, `_decode` (existing in server.py).
- Produces:
  - `POST /learn` (multipart): `file` (rectified image) + form field `payload` (JSON string `{"final_spots": [[x,y],...], "auto_candidates": [[x,y],...]}`). Returns `apply_correction(...)` dict or `{"ok": false, "error": "..."}` (HTTP 200; never 500 on bad input).
  - `GET /model`: returns `learn.model_info()`.

- [ ] **Step 1: Write the failing test (FastAPI TestClient)**

Append to `cv/tests/test_learn.py`:
```python
def test_endpoints_learn_and_model(tmp_path, monkeypatch):
    monkeypatch.setattr(learn, "CLF_PATH", tmp_path / "clf.pkl")
    monkeypatch.setattr(learn, "SAMPLES_PATH", tmp_path / "s.npz")
    from fastapi.testclient import TestClient
    from chromalog_cv.server import app
    import cv2, json
    client = TestClient(app)

    assert client.get("/model").json()["trained"] is False

    img = np.zeros((200, 200, 3), np.uint8)
    ok, enc = cv2.imencode(".png", img)
    payload = json.dumps({"final_spots": [[0.5, 0.5]],
                          "auto_candidates": [[0.5, 0.5], [0.05, 0.05]]})
    r = client.post("/learn",
                    files={"file": ("p.png", enc.tobytes(), "image/png")},
                    data={"payload": payload})
    body = r.json()
    assert body["ok"] is True and body["batch"]["pos"] >= 1
    assert client.get("/model").json()["trained"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /Users/bruceli/Documents/hackathon/cv && .venv/bin/python -m pytest tests/test_learn.py -k endpoints -v
```
Expected: FAIL — 404 / route missing (`/learn`).

- [ ] **Step 3: Write minimal implementation**

In `cv/chromalog_cv/server.py`:

(a) Add imports near the top group: `import json` and `from fastapi import Form`, and `from . import learn as LN`.

(b) Add endpoints after the `/rectify` route:
```python
@app.post("/learn")
async def learn_endpoint(
    file: UploadFile = File(...),
    payload: str = Form(...),
):
    """从一次手动矫正在线增量训练斑点分类器 (设计 §3.2)。
    payload: {"final_spots": [[x,y]...], "auto_candidates": [[x,y]...]} 归一化质心。
    任何坏输入返回 ok:false, 不抛 500。"""
    try:
        img = _decode(await file.read())
        data = json.loads(payload)
        final_pts = [(float(p[0]), float(p[1])) for p in data.get("final_spots", [])]
        auto_pts = [(float(p[0]), float(p[1])) for p in data.get("auto_candidates", [])]
    except Exception as e:
        return JSONResponse(status_code=200, content={"ok": False, "error": str(e)})
    try:
        return LN.apply_correction(img, final_pts, auto_pts, Config())
    except Exception as e:
        return JSONResponse(status_code=200, content={"ok": False, "error": str(e)})


@app.get("/model")
def model_endpoint():
    return LN.model_info()
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cd /Users/bruceli/Documents/hackathon/cv && .venv/bin/python -m pytest tests/ -v
```
Expected: PASS (13 passed).

- [ ] **Step 5: Restart sidecar and smoke-test endpoints live**

Run:
```bash
cd /Users/bruceli/Documents/hackathon/cv
pkill -f chromalog_cv.server 2>/dev/null; sleep 1
nohup .venv/bin/python3 -m chromalog_cv.server >/tmp/chromalog_sidecar.log 2>&1 &
sleep 3; curl -s http://127.0.0.1:8765/model
```
Expected: JSON `{"trained": false, "n_samples": 0, "updated_at": null}` (or trained:true if a model exists in cv/models/).

- [ ] **Step 6: Commit**

```bash
cd /Users/bruceli/Documents/hackathon
git add cv/chromalog_cv/server.py cv/tests/test_learn.py
git commit -m "feat(learn): POST /learn + GET /model sidecar endpoints"
```

---

## Task 10: Swift CVClient — learn() + modelInfo()

**Files:**
- Modify: `App/Sources/ChromaLog/CVClient.swift`

**Interfaces:**
- Consumes: running sidecar `/learn`, `/model`.
- Produces:
  - `struct CVModelInfo: Decodable { let trained: Bool; let n_samples: Int; let updated_at: String? }`
  - `func modelInfo() async -> CVModelInfo?` (nil on any failure).
  - `func learn(rectified: Data, finalSpots: [CGPoint], autoCandidates: [CGPoint]) async` — best-effort; swallows errors.

- [ ] **Step 1: Add response struct + methods**

In `App/Sources/ChromaLog/CVClient.swift`, after `CVRectifyResponse` add:
```swift
struct CVModelInfo: Decodable {
    let trained: Bool
    let n_samples: Int
    let updated_at: String?
}
```
At the top add `import CoreGraphics` if not present (CGPoint). Then inside `final class CVClient`, after `rectify(...)`, add:
```swift
func modelInfo() async -> CVModelInfo? {
    guard let url = URL(string: "\(baseURL)/model") else { return nil }
    do {
        let (data, resp) = try await session.data(from: url)
        guard (resp as? HTTPURLResponse)?.statusCode == 200 else { return nil }
        return try JSONDecoder().decode(CVModelInfo.self, from: data)
    } catch { return nil }
}

/// Best-effort online learning from one manual correction. Errors are ignored.
func learn(rectified: Data, finalSpots: [CGPoint], autoCandidates: [CGPoint]) async {
    guard let url = URL(string: "\(baseURL)/learn") else { return }
    let payload: [String: Any] = [
        "final_spots": finalSpots.map { [$0.x, $0.y] },
        "auto_candidates": autoCandidates.map { [$0.x, $0.y] },
    ]
    guard let payloadData = try? JSONSerialization.data(withJSONObject: payload) else { return }
    let payloadStr = String(data: payloadData, encoding: .utf8) ?? "{}"

    var request = URLRequest(url: url)
    request.httpMethod = "POST"
    let boundary = UUID().uuidString
    request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
    var body = Data()
    // image part
    body.append("--\(boundary)\r\n".data(using: .utf8)!)
    body.append("Content-Disposition: form-data; name=\"file\"; filename=\"rect.png\"\r\n".data(using: .utf8)!)
    body.append("Content-Type: image/png\r\n\r\n".data(using: .utf8)!)
    body.append(rectified)
    body.append("\r\n".data(using: .utf8)!)
    // payload field
    body.append("--\(boundary)\r\n".data(using: .utf8)!)
    body.append("Content-Disposition: form-data; name=\"payload\"\r\n\r\n".data(using: .utf8)!)
    body.append(payloadStr.data(using: .utf8)!)
    body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)
    request.httpBody = body
    _ = try? await session.data(for: request)
}
```

- [ ] **Step 2: Build**

Run:
```bash
cd /Users/bruceli/Documents/hackathon/App && swift build 2>&1 | tail -20
```
Expected: `Build complete!` (or only pre-existing warnings).

- [ ] **Step 3: Commit**

```bash
cd /Users/bruceli/Documents/hackathon
git add App/Sources/ChromaLog/CVClient.swift
git commit -m "feat(app): CVClient.learn + modelInfo"
```

---

## Task 11: AppStore — capture candidates, learn on save, expose count

**Files:**
- Modify: `App/Sources/ChromaLog/AppStore.swift:303-340` (applyDetection), `:418-422` (saveCurrentPlate tail), and the published-properties area near line 78-104.

**Interfaces:**
- Consumes: `CVClient.learn`, `CVClient.modelInfo` (Task 10).
- Produces: `@Published var autoCandidates: [CGPoint]`, `@Published var modelTrainedCount: Int`, `func refreshModelInfo()`.

- [ ] **Step 1: Add published properties**

In `App/Sources/ChromaLog/AppStore.swift`, near the other `@Published` detection props (around line 78), add:
```swift
    /// Snapshot of auto-detected spot centroids (before user edits) — training negatives source.
    @Published var autoCandidates: [CGPoint] = []
    /// Number of correction samples the engine has learned from (for the inspector status line).
    @Published var modelTrainedCount: Int = 0
```

- [ ] **Step 2: Capture candidates in applyDetection**

In `applyDetection(_:)`, inside the `if !spotsUserModified {` block right after the loop that appends spots (after line ~330), the snapshot must capture ALL detected candidates regardless of the edit flag. Add immediately after the `lastEngineUsed = result.engine_used` line:
```swift
        autoCandidates = result.spots.map { CGPoint(x: $0.x, y: $0.y) }
```

- [ ] **Step 3: Learn after a successful save**

In `saveCurrentPlate()`, after `currentExperimentID = id` (line ~419) and within the `do` block where the save succeeded, add a best-effort learning call. Insert after the line that sets `saveStatus = "Saved \(record.title)"`:
```swift
            // Online learning from this correction (best-effort; never blocks saving).
            if let rectified = plateImage, let rectData = rectified.pngData() {
                let finals = spots.map { $0.point }
                let cands = autoCandidates
                Task.detached {
                    await CVClient.shared.learn(rectified: rectData,
                                                finalSpots: finals, autoCandidates: cands)
                    if let info = await CVClient.shared.modelInfo() {
                        await MainActor.run { self.modelTrainedCount = info.n_samples }
                    }
                }
            }
```
(If `NSImage` lacks `pngData()`, reuse the existing `jpegData()` helper and change the part filename/Content-Type in `learn` to jpeg — but prefer PNG to avoid recompression. Check `App/Sources/ChromaLog/*` for an existing `pngData()`/`jpegData()` NSImage extension; `jpegData()` is already used in `runAutoDetect`.)

- [ ] **Step 4: Add refreshModelInfo()**

Add a method near `runAutoDetect`:
```swift
    func refreshModelInfo() {
        Task { @MainActor in
            if let info = await CVClient.shared.modelInfo() {
                self.modelTrainedCount = info.n_samples
            }
        }
    }
```

- [ ] **Step 5: Build**

Run:
```bash
cd /Users/bruceli/Documents/hackathon/App && swift build 2>&1 | tail -20
```
Expected: `Build complete!`. If `pngData()` is undefined, switch to `jpegData()` per Step 3 note and rebuild.

- [ ] **Step 6: Commit**

```bash
cd /Users/bruceli/Documents/hackathon
git add App/Sources/ChromaLog/AppStore.swift
git commit -m "feat(app): snapshot auto-candidates, learn on save, track trained count"
```

---

## Task 12: Inspector status line + manual end-to-end verification

**Files:**
- Modify: `App/Sources/ChromaLog/Views/InspectorView.swift`

**Interfaces:**
- Consumes: `store.modelTrainedCount`, `store.refreshModelInfo()` (Task 11).

- [ ] **Step 1: Add the status line**

In `App/Sources/ChromaLog/Views/InspectorView.swift`, find where engine/detection status is shown (search for `lastEngineUsed` or the detection section). Add a row that renders only when trained:
```swift
            if store.modelTrainedCount > 0 {
                Text("Learned from \(store.modelTrainedCount) corrections")
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
            }
```
And ensure it refreshes on appear — add to the view's `.onAppear` (or the top-level container's), or the existing one:
```swift
            .onAppear { store.refreshModelInfo() }
```
(If an `.onAppear` already exists, add the `store.refreshModelInfo()` call inside it rather than adding a second modifier.)

- [ ] **Step 2: Build**

Run:
```bash
cd /Users/bruceli/Documents/hackathon/App && swift build 2>&1 | tail -20
```
Expected: `Build complete!`

- [ ] **Step 3: Manual end-to-end verification**

1. Ensure the sidecar runs the latest code:
```bash
cd /Users/bruceli/Documents/hackathon/cv
pkill -f chromalog_cv.server 2>/dev/null; sleep 1
nohup .venv/bin/python3 -m chromalog_cv.server >/tmp/chromalog_sidecar.log 2>&1 &
sleep 2; curl -s http://127.0.0.1:8765/model
```
2. Launch the app:
```bash
cd /Users/bruceli/Documents/hackathon/App && swift run 2>&1 | tail -5
```
3. In the app: import a plate (e.g. `training_pictures/TLC_real_2.jpg`), Auto-detect, delete any false-positive spots (corners/glare), add any missed real spots, then Save. Confirm:
   - Inspector shows `Learned from N corrections` (N > 0).
   - `curl -s http://127.0.0.1:8765/model` → `trained: true`, `n_samples` grew.
4. Re-import the SAME plate and Auto-detect again. Confirm `engine_used` ends with `+skl` (check `store.lastEngineUsed` via the UI) and the previously-deleted false positives are no longer detected.

- [ ] **Step 4: Run the full Python suite once more (regression)**

Run:
```bash
cd /Users/bruceli/Documents/hackathon/cv && .venv/bin/python -m pytest tests/ -v
```
Expected: PASS (13 passed).

- [ ] **Step 5: Commit**

```bash
cd /Users/bruceli/Documents/hackathon
git add App/Sources/ChromaLog/Views/InspectorView.swift
git commit -m "feat(app): inspector 'Learned from N corrections' status line"
```

---

## Self-Review

**Spec coverage:**
- design §2 data flow → Tasks 4-9, 11 (derive → learn → detect). ✓
- design §3.1 learn.py (features/classifier/derive/persist) → Tasks 2-5. ✓
- design §3.2 `/learn` + `/model` → Task 9. ✓
- design §4 three-tier precedence (clf > AI > area-knee), authoritative scorer → Tasks 6-7. ✓
- design §5 app changes (autoCandidates, learn on save, no new toggle, inspector line) → Tasks 10-12. ✓
- design §6 models/ pickle, gitignore, YOLO seam → Task 1 (gitignore), Tasks 3/7 (pickle), seam documented in learn.py header + proposer/scorer split. ✓
- design §7 scikit-learn only → Task 1. ✓
- design §8 tests (unit + e2e + regression) → Tasks 2-9 unit, Task 8 e2e, Tasks 8/12 regression. ✓
- design §9 risks (overfit/imbalance/alignment) → easy-neg balancing (Task 4), authoritative-scorer note (Task 6), rectified-image alignment (Task 11 uses plateImage). ✓

**Placeholder scan:** No TBD/TODO; all code steps contain complete code. The one conditional ("if pngData() is undefined, use jpegData()") gives an exact fallback, not a placeholder.

**Type consistency:** `patch_features(img, cx, cy, cfg)` used identically in Tasks 2/4/7. `SpotClassifier.update/proba/save/load/is_trained/n_samples` consistent across Tasks 3/5/7. `apply_correction`/`model_info` signatures match between Tasks 5 and 9. `detect_spots(..., keep_regions, scorer)` consistent between Tasks 6 and 7. Swift `learn(rectified:finalSpots:autoCandidates:)` / `modelInfo()` consistent between Tasks 10 and 11.
