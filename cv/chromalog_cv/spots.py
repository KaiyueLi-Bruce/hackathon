"""⑤ 斑点候选检测 + ⑥ 泳道归组 + Rf 计算 (附录 D.2 / 第 7 节)。

⑤ 连通域 -> 按面积/填充度/长宽比/位置过滤 (排除长横线=基线/前沿, 排除噪点/污渍)。
⑥ 候选 x 坐标投影成直方图找峰 -> 切泳道 -> 每个候选归属一列。
Rf = (baselineY - spotY) / (baselineY - frontY)   (图像 y 向下, 基线在下、前沿在上)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import Config


@dataclass
class Spot:
    x: float          # 强度加权质心 x (像素, 校正图)
    y: float          # 强度加权质心 y
    bbox: Tuple[int, int, int, int]  # x,y,w,h
    area: int
    lane: int
    rf: Optional[float]
    shape: str = "blob"  # round | elliptical | tailing | irregular


@dataclass
class SpotsResult:
    spots: List[Spot] = field(default_factory=list)
    lane_bounds: List[Tuple[int, int]] = field(default_factory=list)  # 每泳道 x 范围


def _rf(spot_y: float, baseline_y: Optional[float], front_y: Optional[float]) -> Optional[float]:
    if baseline_y is None or front_y is None or baseline_y == front_y:
        return None
    return float((baseline_y - spot_y) / (baseline_y - front_y))


def _area_knee_cut(spots: List[Spot], cfg: Config) -> List[Spot]:
    """尽量少标注: 把候选按面积由大到小排, 对 log(面积) 随排名做线性回归,
    预测下一个的应有面积; 当下一个实际面积 << 预测 (严重偏离, 即面积曲线的"悬崖")
    时停止, 其后更小的一律视为噪点丢弃。

    直觉: 真实斑点面积大致同量级、缓降; 噪点会出现一个陡降的断崖。
    """
    if not cfg.knee_enabled or len(spots) <= cfg.knee_min_points:
        return sorted(spots, key=lambda s: -s.area)

    ranked = sorted(spots, key=lambda s: -s.area)
    logs = np.log(np.array([s.area for s in ranked], dtype=np.float64) + 1.0)

    keep = list(ranked[:cfg.knee_min_points])
    drop_log = np.log(cfg.knee_deviation)  # 预测-实际 超过此值即"严重偏离"
    for k in range(cfg.knee_min_points, len(ranked)):
        x = np.arange(k)
        a, b = np.polyfit(x, logs[:k], 1)   # 对已保留的点拟合直线
        pred = a * k + b                     # 预测第 k 个 (下一个) 的 log 面积
        if logs[k] < pred - drop_log:        # 实际远小于预测 -> 断崖, 停止
            break
        keep.append(ranked[k])

    return keep if len(keep) >= cfg.knee_min_keep else ranked[:cfg.knee_min_keep]


def _filter_by_regions(spots: List[Spot], regions, plate_w: int, plate_h: int,
                       cfg: Config) -> List[Spot]:
    """AI 精修: 只保留质心落在 LLM 粗框(归一化)内的候选 (框外扩一点容错)。
    regions: [{x,y,w,h,...}] 归一化。LLM 只给'哪里有真斑点', 几何仍是 OpenCV 的。"""
    if not regions:
        return []
    pad = 0.02
    kept = []
    for s in spots:
        nx, ny = s.x / plate_w, s.y / plate_h
        for r in regions:
            if (r["x"] - pad <= nx <= r["x"] + r["w"] + pad and
                    r["y"] - pad <= ny <= r["y"] + r["h"] + pad):
                kept.append(s)
                break
    return kept


def _assign_lanes(spots: List[Spot], plate_w: int, cfg: Config) -> List[Tuple[int, int]]:
    if not spots:
        return []
    xs = np.array([s.x for s in spots])
    hist = np.zeros(plate_w, np.float32)
    for x in xs:
        hist[int(np.clip(x, 0, plate_w - 1))] += 1
    win = max(1, int(cfg.lane_smooth_frac * plate_w)) | 1  # 奇数
    hist = cv2.GaussianBlur(hist.reshape(-1, 1), (1, win), 0).ravel()

    # 找峰: 局部极大且间距 >= 最小间隔
    min_gap = int(cfg.lane_min_gap_frac * plate_w)
    peaks = []
    for i in range(1, plate_w - 1):
        if hist[i] > 0 and hist[i] >= hist[i - 1] and hist[i] >= hist[i + 1]:
            if not peaks or i - peaks[-1] >= min_gap:
                peaks.append(i)
            elif hist[i] > hist[peaks[-1]]:
                peaks[-1] = i
    if not peaks:
        peaks = [int(np.median(xs))]

    # 泳道边界 = 相邻峰中点
    bounds = []
    for j, p in enumerate(peaks):
        lo = 0 if j == 0 else (peaks[j - 1] + p) // 2
        hi = plate_w if j == len(peaks) - 1 else (p + peaks[j + 1]) // 2
        bounds.append((lo, hi))

    for s in spots:
        s.lane = min(range(len(bounds)),
                     key=lambda j: abs(s.x - (bounds[j][0] + bounds[j][1]) / 2))
    return bounds


def _classify_shape(w: int, h: int, extent: float, cfg: Config) -> str:
    elong = max(w, h) / max(1, min(w, h))
    if h >= w and elong >= cfg.tailing_elong:
        return "tailing"          # 纵向拉长 -> 拖尾 (仍是合法斑点)
    if elong < 1.3 and extent >= 0.6:
        return "round"
    if extent >= 0.45:
        return "elliptical"
    return "irregular"


def _is_line(w: int, h: int, plate_w: int, plate_h: int, cfg: Config) -> bool:
    """剔除'线/条': 跨度过大者非斑点 (真斑点不会横贯板宽或纵贯板高)。
    按 span 单独判定 —— 不再要求'很细', 故 9px 宽却高达 0.7 板高的边条也能被剔。"""
    return (w >= cfg.line_span_frac * plate_w) or (h >= cfg.line_span_frac * plate_h)


def detect_spots(binary: np.ndarray, gray: np.ndarray, polarity: str, roi,
                 baseline_y, front_y, plate_area: float, cfg: Config,
                 keep_regions=None) -> SpotsResult:
    x0, y0, x1, y1 = roi
    plate_h, plate_w = binary.shape[:2]
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    # 斑点强度图: 暗点亮底 -> 越暗越强; 亮点暗底 -> 越亮越强。用于强度加权质心(拖尾更准)。
    g = gray.astype(np.float32)
    intensity = (255.0 - g) if polarity == "dark_on_light" else g

    spots: List[Spot] = []
    min_a = cfg.spot_min_area_frac * plate_area
    max_a = cfg.spot_max_area_frac * plate_area
    for i in range(1, n):  # 0 是背景
        x, y, w, h, area = stats[i]
        if area < min_a or area > max_a:
            continue
        if _is_line(w, h, plate_w, plate_h, cfg):
            continue  # 形状无关: 仅排除线, 不排除非圆斑点
        extent = area / float(max(1, w * h))
        if extent < cfg.spot_min_extent:
            continue  # 极低填充度: 碎屑/空心噪声
        # 强度加权质心 (在该连通域 mask 内)
        comp = (labels[y:y + h, x:x + w] == i)
        wts = intensity[y:y + h, x:x + w] * comp
        s = float(wts.sum())
        if s <= 0:
            cx, cy = x + w / 2.0, y + h / 2.0
        else:
            ys_, xs_ = np.nonzero(comp)
            cx = x + float((xs_ * wts[ys_, xs_]).sum() / s)
            cy = y + float((ys_ * wts[ys_, xs_]).sum() / s)
        if not (x0 <= cx <= x1 and y0 <= cy <= y1):
            continue  # ROI 之外
        emx, emy = cfg.edge_margin_frac * plate_w, cfg.edge_margin_frac * plate_h
        if cx < emx or cx > plate_w - emx or cy < emy or cy > plate_h - emy:
            continue  # 质心贴近图像边缘 -> 板边/阴影
        shape = _classify_shape(int(w), int(h), extent, cfg)
        spots.append(Spot(cx, cy, (int(x), int(y), int(w), int(h)),
                          int(area), -1, _rf(cy, baseline_y, front_y), shape))

    # ⑤b 噪点过滤: 有 LLM 粗框则用 AI 区域精修; 否则面积回归拐点压噪
    if keep_regions is not None:
        spots = _filter_by_regions(spots, keep_regions, plate_w, plate_h, cfg)
    else:
        spots = _area_knee_cut(spots, cfg)

    bounds = _assign_lanes(spots, plate_w, cfg)
    spots.sort(key=lambda s: (s.lane, s.y))
    return SpotsResult(spots=spots, lane_bounds=bounds)
