"""② 光照归一化 (附录 D.2)。

UV 灯下拍摄光照极不均匀 (中间亮、四周暗、有辉光), 直接阈值会被光照干扰。
CLAHE 自适应直方图均衡压掉不均匀光照, 让后续二值化稳定。
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import Config


def to_gray_clahe(bgr: np.ndarray, cfg: Config) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip,
                            tileGridSize=(cfg.clahe_grid, cfg.clahe_grid))
    return clahe.apply(gray)
