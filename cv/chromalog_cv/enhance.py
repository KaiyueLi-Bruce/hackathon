"""扫描件增强 (CamScanner 风格)。

借鉴 ArashNasrEsfahani/Python-Document-Scanner-OpenCV 的增强阶段:
  gamma 校正 -> LAB 的 L 通道做 CLAHE -> 饱和度提升 -> 反锐化掩膜。
让正畸后的板看起来像一张干净、清晰的"扫描件"。

注意: 仅用于**显示**的正畸图; 斑点检测/Rf 计算仍跑在未增强的几何图上,
以免饱和度/锐化改变斑点强度与几何。
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import Config


def enhance_scan(bgr: np.ndarray, cfg: Config) -> np.ndarray:
    out = bgr

    # 1) gamma 校正 (提亮中间调)
    g = max(0.05, float(cfg.enhance_gamma))
    lut = (((np.arange(256) / 255.0) ** (1.0 / g)) * 255.0).clip(0, 255).astype(np.uint8)
    out = cv2.LUT(out, lut)

    # 2) LAB 的 L 通道 CLAHE (局部对比, 不放大彩噪)
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=cfg.enhance_clahe_clip, tileGridSize=(8, 8))
    l = clahe.apply(l)
    out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # 3) 饱和度提升
    if abs(cfg.enhance_saturation - 1.0) > 1e-3:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * cfg.enhance_saturation, 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 4) 反锐化掩膜 (提升清晰度)
    amt = float(cfg.enhance_sharpen)
    if amt > 1e-3:
        blur = cv2.GaussianBlur(out, (0, 0), 3)
        out = cv2.addWeighted(out, 1.0 + amt, blur, -amt, 0)

    return out
