"""③ 自动极性二值化 / 斑点分割 (附录 D.2)。

用 top-hat / black-hat 形态学做斑点检测, 而非全局阈值:
  - black-hat = 闭运算(gray) − gray  -> 凸显"比周围暗、且小于核"的区域 (暗点亮底, UV254/显色剂)
  - top-hat   = gray − 开运算(gray)   -> 凸显"比周围亮、且小于核"的区域 (亮点暗底, UV365)
hat 变换天然消除均匀背景与缓变光照, 并压制细密纹理噪点 (噪点幅度小, 阈值即滤掉)。

极性自动判定: 两种 hat 在 ROI 内的强响应 (99 分位) 谁大即谁; 接近则判"不确定"。
阈值 = 均值 + k·标准差 (k 越大越保守, 对应"尽量少标注")。

前置: 须先 ① 透视校正; hat 本身是局部对比算子, 对不均匀光照鲁棒, 不强依赖 CLAHE。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import Config


@dataclass
class BinarizeResult:
    binary: np.ndarray              # uint8, 斑点=255
    gray: np.ndarray               # 处理用灰度 (供质心/强度加权)
    roi: Tuple[int, int, int, int]  # x0,y0,x1,y1 有效区
    polarity: str                  # "dark_on_light" | "light_on_dark"
    minority_frac: float           # 斑点像素在 ROI 内占比 (仅供参考/告警)
    uncertain: bool


def _roi_from_lines(shape, baseline_y, front_y, cfg: Config) -> Tuple[int, int, int, int]:
    h, w = shape[:2]
    inset_x = int(cfg.roi_inset_frac * w)
    inset_y = int(cfg.roi_inset_frac * h)
    y_top = int(front_y) if front_y is not None else inset_y
    y_bot = int(baseline_y) if baseline_y is not None else h - inset_y
    y_top = max(inset_y, min(y_top, h - 1))
    y_bot = max(y_top + 1, min(y_bot, h - inset_y))
    # 防塌缩: ROI 过矮 (基线/前沿异常) -> 退回整板有效区
    if y_bot - y_top < 0.10 * h:
        y_top, y_bot = inset_y, h - inset_y
    return (inset_x, y_top, w - inset_x, y_bot)


def auto_binarize(bgr: np.ndarray, baseline_y: Optional[float],
                  front_y: Optional[float], cfg: Config) -> BinarizeResult:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    h, w = gray.shape[:2]
    x0, y0, x1, y1 = _roi_from_lines(gray.shape, baseline_y, front_y, cfg)

    roi_mask = np.zeros((h, w), np.uint8)
    roi_mask[y0:y1, x0:x1] = 1

    # 极性判定: 用 Otsu 少数派 (鲁棒) —— 斑点是少数派。
    roi_px = gray[y0:y1, x0:x1]
    thr_o, _ = cv2.threshold(roi_px, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark_a = int((roi_px < thr_o).sum())
    light_a = int((roi_px >= thr_o).sum())
    total = max(1, dark_a + light_a)
    if dark_a <= light_a:
        polarity, minority_dec = "dark_on_light", dark_a / total
    else:
        polarity, minority_dec = "light_on_dark", light_a / total
    uncertain = cfg.polarity_uncertain_lo <= minority_dec <= cfg.polarity_uncertain_hi

    # 斑点掩膜: 按极性选对应 hat (暗点->black-hat; 亮点->top-hat)
    ksz = max(3, int(cfg.hat_kernel_frac * min(w, h)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
    op = cv2.MORPH_BLACKHAT if polarity == "dark_on_light" else cv2.MORPH_TOPHAT
    resp = cv2.morphologyEx(gray, op, kernel)

    # 阈值 = 均值 + k·标准差 (仅在 ROI 内统计), 保守
    vals = resp[y0:y1, x0:x1]
    m, s = float(vals.mean()), float(vals.std())
    thr = m + cfg.hat_thresh_k * s
    binary = ((resp >= thr).astype(np.uint8) * 255)
    binary = cv2.bitwise_and(binary, binary, mask=roi_mask * 255)

    # 形态学清理: 开运算去残余斑驳, 闭运算补斑点
    k = max(1, int(cfg.morph_frac * h))
    mk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, mk, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, mk, iterations=1)

    roi_area = max(1, int(roi_mask.sum()))
    minority = float((binary > 0).sum()) / roi_area
    return BinarizeResult(binary, gray, (x0, y0, x1, y1), polarity, minority, bool(uncertain))
