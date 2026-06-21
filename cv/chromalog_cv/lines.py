"""④ 自动识别铅笔基线 / 溶剂前沿 (附录 D.2)。

默认约定: 基线与溶剂前沿两条线均由用户用铅笔画出 (细、暗、近水平、横贯板宽)。
HoughLinesP 检测 -> 过滤(近水平 + 够长) -> 按 y 合并 -> 通常得两条:
下线(y 大)=基线(原点), 上线(y 小)=溶剂前沿。

兜底 (仅当少于两条时): 只检到一条 -> 视为基线, 前沿由干湿分界估计;
一条都没有 -> 全交用户事后拖动。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

import cv2
import numpy as np

from .config import Config


@dataclass
class LinesResult:
    baseline_y: Optional[float]   # 像素 y (校正图坐标), 下方
    front_y: Optional[float]      # 像素 y, 上方
    detected: List[float]         # 检到的所有横线 y
    baseline_from: str            # "hough" | "fallback" | "none"
    front_from: str               # "hough" | "wetfront" | "none"


def _hough_horizontal(gray: np.ndarray, cfg: Config) -> List[float]:
    h, w = gray.shape[:2]
    v = float(np.median(gray))
    edges = cv2.Canny(gray, int(max(0, 0.66 * v)), int(min(255, 1.33 * v)))
    min_len = int(cfg.line_min_len_frac * w)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=max(40, min_len // 3),
                            minLineLength=min_len, maxLineGap=int(0.05 * w))
    ys: List[float] = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0, :]:
            ang = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            ang = min(ang, 180 - ang)
            if ang <= cfg.line_max_angle_deg:
                ys.append((y1 + y2) / 2.0)
    return ys


def _merge_ys(ys: List[float], h: int, cfg: Config) -> List[float]:
    if not ys:
        return []
    ys = sorted(ys)
    tol = cfg.line_merge_frac * h
    groups = [[ys[0]]]
    for y in ys[1:]:
        if y - groups[-1][-1] <= tol:
            groups[-1].append(y)
        else:
            groups.append([y])
    # 取每组均值, 按"出现票数"降序保留最显著的几条
    merged = sorted(groups, key=len, reverse=True)
    return [float(np.mean(g)) for g in merged]


def _wet_front_estimate(gray: np.ndarray, baseline_y: Optional[float]) -> Optional[float]:
    """干湿分界估计: 行均值在板上半部的最大梯度处 (湿区与未展开区亮度突变)。"""
    h = gray.shape[0]
    prof = gray.mean(axis=1).astype(np.float32)
    prof = cv2.GaussianBlur(prof.reshape(-1, 1), (1, 9), 0).ravel()
    grad = np.abs(np.gradient(prof))
    top = int(0.05 * h)
    bot = int((baseline_y if baseline_y else 0.9 * h))
    bot = min(bot, h - 1)
    if bot - top < 5:
        return None
    idx = int(np.argmax(grad[top:bot]) + top)
    return float(idx)


def _detect_lines_learned(gray, cfg, line_clf):
    """已学线模型: 在候选行(Hough + 网格扫描)里按 P(是线) 选下半=基线、上半=前沿。"""
    from . import learn as LN
    h = gray.shape[0]
    hough = _merge_ys(_hough_horizontal(gray, cfg), h, cfg)
    grid = [i / 100.0 for i in range(5, 96, 2)]
    cand = sorted(set([round(v / h, 3) for v in hough] + grid))
    scores = LN.score_rows(gray, cand, cfg, line_clf)
    lower = [(c, s) for c, s in zip(cand, scores) if c > 0.5]
    upper = [(c, s) for c, s in zip(cand, scores) if c <= 0.5]
    thr = 0.5
    baseline = max(lower, key=lambda t: t[1]) if lower else None
    front = max(upper, key=lambda t: t[1]) if upper else None
    by = baseline[0] * h if baseline and baseline[1] >= thr else None
    fy = front[0] * h if front and front[1] >= thr else None
    return LinesResult(by, fy, [v for v in (by, fy) if v is not None],
                       "learned" if by is not None else "none",
                       "learned" if fy is not None else "none")


def detect_lines(gray: np.ndarray, cfg: Config, line_clf=None) -> LinesResult:
    h, _ = gray.shape[:2]
    # 已学线模型优先 (把'线的识别'纳入学习)
    if line_clf is not None and getattr(line_clf, "is_trained", False):
        res = _detect_lines_learned(gray, cfg, line_clf)
        if res.baseline_y is not None or res.front_y is not None:
            return res
    ys = _merge_ys(_hough_horizontal(gray, cfg), h, cfg)

    if len(ys) >= 2:
        ys_sorted = sorted(ys[:6])  # 取最显著的若干, 再按位置
        baseline = max(ys_sorted)   # y 最大 = 最低 = 基线
        front = min(ys_sorted)      # y 最小 = 最高 = 前沿
        # 防塌缩: 基线/前沿相隔过近 (多为同一条边缘的双线) -> 视为无效, 降级到单线/无
        if baseline - front >= cfg.line_min_sep_frac * h:
            return LinesResult(baseline, front, ys, "hough", "hough")
        ys = [float(np.mean(ys_sorted))]  # 合成一条, 走下方单线逻辑按位置判角色

    if len(ys) == 1:
        y = ys[0]
        # 单条线按位置判角色: 在板上半部 -> 溶剂前沿; 下半部 -> 基线。
        # (无脑当基线会把顶部前沿线误判, 导致 ROI 塌缩。)
        if y < 0.5 * h:
            # 这条是前沿; 基线未知, 交用户事后拖动 (ROI 取前沿下方至板底)
            return LinesResult(None, y, ys, "none", "hough")
        else:
            # 这条是基线; 前沿尝试干湿分界估计
            front = _wet_front_estimate(gray, y)
            return LinesResult(y, front, ys, "hough",
                               "wetfront" if front is not None else "none")

    return LinesResult(None, None, ys, "none", "none")
