"""① 自动找板 + 透视校正 (附录 D.2)。

掩膜来源可插拔:
  - 来源 A: OpenCV 边缘/阈值 (本文件, 零训练) —— 默认。
  - 来源 B: 文档分割模型 U-Net/DeepLab -> ONNX (M5+ 升级)。
    只需实现 plate_mask(bgr) -> uint8 mask, 下游 "掩膜->四角->warp" 完全共用。

降级: 找不到可信四边形 -> 返回原图, rectified=False, 低置信度;
绝不因一张难图整链崩溃。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .config import Config


@dataclass
class RectifyResult:
    image: np.ndarray          # 校正后 (或原图) BGR
    rectified: bool            # 是否成功做了透视校正
    confidence: float          # 0..1
    quad: Optional[np.ndarray] # 原图坐标系下的四角 (4,2) float32, 失败为 None
    note: str = ""


# ---------- 掩膜来源 A: OpenCV ----------

def plate_mask_opencv(bgr: np.ndarray, cfg: Config) -> np.ndarray:
    """用边缘 + 形态学闭合得到板子前景掩膜 (uint8 0/255)。"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # 自适应 Canny: 阈值由中位数推导, 适应不同曝光
    v = float(np.median(gray))
    lo = int(max(0, 0.66 * v))
    hi = int(min(255, 1.33 * v))
    edges = cv2.Canny(gray, lo, hi)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=2)
    edges = cv2.dilate(edges, k, iterations=1)
    return edges


def _order_quad(pts: np.ndarray) -> np.ndarray:
    """把 4 点排成 [左上, 右上, 右下, 左下]。"""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)            # x + y
    d = np.diff(pts, axis=1).ravel()  # y - x
    return np.array([
        pts[np.argmin(s)],   # 左上: x+y 最小
        pts[np.argmin(d)],   # 右上: y-x 最小 (x 大 y 小)
        pts[np.argmax(s)],   # 右下: x+y 最大
        pts[np.argmax(d)],   # 左下: y-x 最大 (x 小 y 大)
    ], dtype=np.float32)


def find_plate_quad(mask: np.ndarray, img_area: float, cfg: Config):
    """从掩膜找最大且足够"矩形"的四边形。返回 (quad|None, confidence)。"""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 0.0
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < cfg.plate_min_area_frac * img_area:
            break  # 后面更小, 直接停
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, cfg.plate_quad_eps_frac * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            # 置信度: 轮廓面积 / 其外接框面积 (越接近 1 越像规整矩形)
            rect_area = cv2.minAreaRect(approx)[1]
            rect_area = max(1.0, rect_area[0] * rect_area[1])
            rectangularity = float(area / rect_area)
            coverage = float(area / img_area)
            conf = max(0.0, min(1.0, 0.5 * rectangularity + 0.5 * min(1.0, coverage / 0.8)))
            return approx, conf
    return None, 0.0


def four_point_warp(bgr: np.ndarray, quad: np.ndarray) -> np.ndarray:
    src = _order_quad(quad)
    (tl, tr, br, bl) = src
    w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    w, h = max(w, 1), max(h, 1)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(bgr, M, (w, h))


def plate_mask_otsu(bgr: np.ndarray, cfg: Config) -> np.ndarray:
    """前景掩膜: 板子通常与背景有整体亮度/颜色差 (UV 绿光板尤其明显)。
    Otsu 分前景背景, 取占据中心、面积较大的一类为板。"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 让"中心区域占多数"的一类作为前景 (板一般在画面中部)
    h, w = th.shape
    cy0, cy1, cx0, cx1 = int(0.3 * h), int(0.7 * h), int(0.3 * w), int(0.7 * w)
    center = th[cy0:cy1, cx0:cx1]
    if center.mean() < 127:      # 中心偏暗 -> 前景是暗类, 反相
        th = cv2.bitwise_not(th)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
    return th


def plate_mask_bgcolor(bgr: np.ndarray, cfg: Config) -> np.ndarray:
    """扫描软件套路: 采样四角作为背景色, 与背景色差异大的像素即前景(板)。
    对'板 vs 较均匀背景'(桌面/台面)最稳, 不依赖明暗只看颜色距离。"""
    h, w = bgr.shape[:2]
    img = cv2.GaussianBlur(bgr, (5, 5), 0).astype(np.float32)
    s = max(8, int(0.06 * min(h, w)))
    corners = np.concatenate([
        img[:s, :s].reshape(-1, 3), img[:s, -s:].reshape(-1, 3),
        img[-s:, :s].reshape(-1, 3), img[-s:, -s:].reshape(-1, 3),
    ], axis=0)
    bg_mean = corners.mean(axis=0)
    bg_std = corners.std(axis=0).mean()
    dist = np.linalg.norm(img - bg_mean, axis=2)
    thr = max(35.0, 3.0 * bg_std)
    mask = (dist > thr).astype(np.uint8) * 255
    # 小核清理: 去散点 + 补小洞; 核太大会把噪点桥接成整幅, 误判背景
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, s // 6), max(3, s // 6)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    # 只保留最大连通域 (板), 去掉零散前景
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n > 1:
        big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        mask = (lab == big).astype(np.uint8) * 255
    return mask


def _quads_from_mask(mask: np.ndarray, img_area: float, cfg: Config):
    """从一个掩膜抽取所有'够大的凸四边形'候选 -> [(quad, area, rectangularity)]。"""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:5]:
        area = cv2.contourArea(c)
        if area < cfg.plate_min_area_frac * img_area:
            break
        peri = cv2.arcLength(c, True)
        for eps in (0.02, 0.04, 0.06, 0.08):   # 逐步放宽, 容忍圆角/噪声
            approx = cv2.approxPolyDP(c, eps * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                rect = cv2.minAreaRect(approx)[1]
                ra = max(1.0, rect[0] * rect[1])
                out.append((approx.reshape(4, 2).astype(np.float32), area, float(area / ra)))
                break
    return out


def find_best_quad(small: np.ndarray, img_area: float, cfg: Config):
    """综合多种掩膜(边缘/Otsu/角落背景色)找最佳板四边形。
    选规则: 足够矩形(rectangularity>=0.8) 中取面积最大者。返回 (quad|None, conf)。"""
    cands = []
    for mask in (plate_mask_opencv(small, cfg),
                 plate_mask_otsu(small, cfg),
                 plate_mask_bgcolor(small, cfg)):
        cands += _quads_from_mask(mask, img_area, cfg)
    # 阈值放宽到 0.62: 透视成梯形的板 rectangularity 会偏低(~0.7), 不能因此被拒,
    # 否则只会退回旋转矩形(去旋转不纠透视)。同时排除"整张图边框"那种近满幅四边形(<=0.95)。
    rect_ok = [c for c in cands if c[2] >= 0.62 and c[1] <= 0.95 * img_area]
    if not rect_ok:
        return None, 0.0
    quad, area, rectangularity = max(rect_ok, key=lambda c: c[1])
    conf = max(0.0, min(1.0, 0.5 * rectangularity + 0.5 * min(1.0, area / img_area / 0.7)))
    return quad, conf


def _quad_from_min_area_rect(mask: np.ndarray, img_area: float, cfg: Config):
    """兜底: 取最大前景轮廓的最小外接(旋转)矩形 -> 4 角。
    主要校正旋转/轻微视角, 不要求轮廓恰好 4 点。"""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, 0.0
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < cfg.plate_min_area_frac * img_area:
        return None, 0.0
    box = cv2.boxPoints(cv2.minAreaRect(c))           # (4,2)
    rect_area = max(1.0, cv2.contourArea(box.astype(np.float32)))
    conf = max(0.0, min(1.0, float(area / rect_area)))  # 轮廓越填满外接矩形越可信
    return box.astype(np.float32), conf


def _has_uniform_background(small: np.ndarray, cfg: Config):
    """判断是否'板放在一片均匀背景上'(才需要扫描裁正)。
    条件: 四角颜色均匀(低方差) 且 画面里存在一块明显异于背景的连通区域(=板),
    其占比在合理范围(不太小、也不是整幅)。不假设板居中, 故偏置/角落的板也能识别;
    已裁好的板(整幅=同色, 无异色前景)会判 False, 从而不被破坏。
    返回 (do_scan, plate_frac, corner_std)。"""
    h, w = small.shape[:2]
    img = cv2.GaussianBlur(small, (5, 5), 0).astype(np.float32)
    s = max(8, int(0.06 * min(h, w)))
    corners = np.concatenate([
        img[:s, :s].reshape(-1, 3), img[:s, -s:].reshape(-1, 3),
        img[-s:, :s].reshape(-1, 3), img[-s:, -s:].reshape(-1, 3),
    ], axis=0)
    corner_std = float(corners.std(axis=0).mean())
    # 与四角背景色差异大的最大连通区域 = 板
    plate_frac = float((plate_mask_bgcolor(small, cfg) > 0).mean())
    do_scan = (corner_std < 28.0) and (0.04 <= plate_frac <= 0.95)
    return do_scan, plate_frac, corner_std


def rectify(bgr: np.ndarray, cfg: Config, mask_fn=plate_mask_opencv) -> RectifyResult:
    """主入口。mask_fn 可换成 ONNX 分割 (来源 B)。

    策略 (逐级):
      1) 边缘法找规整四边形 (透视校正最准);
      2) 兜底: Otsu 前景 + 最小外接旋转矩形 (校正旋转/带背景的板);
      3) 仍失败: 用原图 (多为已裁好的板, 无需校正)。
    """
    h0, w0 = bgr.shape[:2]
    scale = min(1.0, cfg.work_max_side / float(max(h0, w0)))
    small = cv2.resize(bgr, None, fx=scale, fy=scale) if scale < 1.0 else bgr
    img_area = small.shape[0] * small.shape[1]

    # 关键守门: 仅当"板放在一片均匀背景上"才做扫描裁正; 已裁好的板原样返回, 绝不破坏。
    do_scan, _, _ = _has_uniform_background(small, cfg)
    if not do_scan:
        return RectifyResult(bgr, False, 0.0, None, "已是裁好的板/无均匀背景, 无需校正")

    # 1) 四点文档扫描: 多掩膜(边缘/Otsu/角落背景色)综合找最佳凸四边形 -> 透视拉正
    quad_s, conf = find_best_quad(small, img_area, cfg)
    if quad_s is not None and conf >= 0.45:
        quad = quad_s / scale
        return RectifyResult(four_point_warp(bgr, quad), True, conf, quad, "透视校正完成(四点扫描)")

    # 2) 兜底: Otsu 前景 + 最小外接旋转矩形 (只去旋转/裁切, 不纠透视)
    quad_s, conf2 = _quad_from_min_area_rect(plate_mask_otsu(small, cfg), img_area, cfg)
    if quad_s is not None and conf2 >= 0.55:
        quad = quad_s.reshape(4, 2).astype(np.float32) / scale
        return RectifyResult(four_point_warp(bgr, quad), True, conf2, quad, "校正完成(旋转矩形)")

    # 3) 降级: 用原图
    return RectifyResult(bgr, False, max(conf, conf2), None,
                         "有背景但未找到可信板边, 用原图继续")


# ---------- 来源 AI: AI 粗定位 + OpenCV 精修四角 ----------

def _denorm_quad(quad_norm, w: int, h: int) -> np.ndarray:
    """归一化四角 -> 像素坐标 (4,2)。"""
    return np.array([[float(x) * w, float(y) * h] for (x, y) in quad_norm],
                    dtype=np.float32)


def _crop_box_from_region(bbox, quad, w: int, h: int, margin: float):
    """从 AI 的归一化 bbox 或 quad 算出带 margin 的轴对齐裁切框 (像素 int)。"""
    if bbox is not None:
        x0n, y0n = bbox["x"], bbox["y"]
        x1n, y1n = x0n + bbox["w"], y0n + bbox["h"]
    else:
        xs = [p[0] for p in quad]
        ys = [p[1] for p in quad]
        x0n, y0n, x1n, y1n = min(xs), min(ys), max(xs), max(ys)
    mw = margin * max(0.0, x1n - x0n)
    mh = margin * max(0.0, y1n - y0n)
    x0 = int(round(max(0.0, x0n - mw) * w))
    y0 = int(round(max(0.0, y0n - mh) * h))
    x1 = int(round(min(1.0, x1n + mw) * w))
    y1 = int(round(min(1.0, y1n + mh) * h))
    return x0, y0, x1, y1


def _quad_valid(q: np.ndarray, w: int, h: int) -> bool:
    """AI 四角合法性: 4 点构成的凸四边形面积须 > 整图 1% (排除退化/重合点)。"""
    if q is None or q.shape != (4, 2):
        return False
    area = abs(cv2.contourArea(_order_quad(q)))
    return area > 0.01 * w * h


def rectify_ai(bgr: np.ndarray, cfg: Config, bbox=None, quad=None,
               margin: float = 0.08) -> RectifyResult:
    """AI 粗定位 + OpenCV 精修四角的正畸 (来源 AI, 见 llm_detect.detect_plate)。

    bbox/quad 为 AI 给的归一化粗略位置。先在该区域内用 OpenCV 精修出精确四角再透视拉正;
    精修失败 -> 退回 AI 四角直接拉正 -> 再退回 AI bbox 轴对齐裁切 -> 最终退回纯 OpenCV。
    与 detect_regions 同样的"AI 粗框 + OpenCV 精修"哲学: 几何精度仍由 OpenCV 保证。
    """
    h0, w0 = bgr.shape[:2]
    if bbox is None and quad is None:
        return rectify(bgr, cfg)

    # ① 在 AI 区域内用 OpenCV 精修四角 (区域内板几乎填满, 阈值天然满足)
    x0, y0, x1, y1 = _crop_box_from_region(bbox, quad, w0, h0, margin)
    if x1 - x0 >= 16 and y1 - y0 >= 16:
        crop = bgr[y0:y1, x0:x1]
        sc = min(1.0, cfg.work_max_side / float(max(crop.shape[:2])))
        small = cv2.resize(crop, None, fx=sc, fy=sc) if sc < 1.0 else crop
        area = small.shape[0] * small.shape[1]
        quad_s, conf = find_best_quad(small, area, cfg)
        if quad_s is not None and conf >= 0.45:
            q = quad_s / sc
            q[:, 0] += x0
            q[:, 1] += y0
            return RectifyResult(four_point_warp(bgr, q), True, conf, q,
                                 "AI 定位 + OpenCV 精修四角")

    # ② 退回: AI 四角直接拉正
    if quad is not None:
        q = _denorm_quad(quad, w0, h0)
        if _quad_valid(q, w0, h0):
            return RectifyResult(four_point_warp(bgr, q), True, 0.5, q,
                                 "AI 四角直接校正 (未精修)")

    # ③ 退回: AI bbox 轴对齐裁切 (无四角时至少裁出板)
    if bbox is not None:
        cx0, cy0, cx1, cy1 = _crop_box_from_region(bbox, None, w0, h0, 0.0)
        if cx1 - cx0 >= 16 and cy1 - cy0 >= 16:
            return RectifyResult(bgr[cy0:cy1, cx0:cx1].copy(), True, 0.4, None,
                                 "AI bbox 裁切 (未精修四角)")

    # ④ 彻底失败 -> 纯 OpenCV
    return rectify(bgr, cfg)
