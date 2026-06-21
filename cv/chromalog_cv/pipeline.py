"""全自动流水线编排 (附录 D.2)。

顺序: ① rectify -> ② CLAHE -> ④ detect_lines(gray) -> ③ auto_binarize
      -> ⑤/⑥ detect_spots。

注: ④ 在 ③ 之前跑 —— Hough 在 gray 上工作, 为 ③ 提供 ROI (基线/前沿之间),
规格里 ③④ 的编号是逻辑列举, 实现上线检测须先行。

输出坐标一律归一化到校正图尺寸 (0..1, y 向下), 交 SwiftUI 渲染;
用户事后拖动基线/前沿即可在前端实时重算 Rf。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any

import cv2
import numpy as np

from .config import Config
from . import rectify as R
from . import preprocess as P
from . import lines as L
from . import binarize as B
from . import spots as S
from . import enhance as E


@dataclass
class PipelineResult:
    width: int
    height: int
    rectified: bool
    rectify_confidence: float
    polarity: str
    polarity_uncertain: bool
    minority_frac: float
    baseline_y_norm: Optional[float]
    front_y_norm: Optional[float]
    baseline_from: str
    front_from: str
    n_lanes: int
    spots: List[Dict[str, Any]]
    warnings: List[str]
    engine_used: str = "opencv"   # opencv | ai+opencv | yolo

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


def run_pipeline(bgr: np.ndarray, cfg: Optional[Config] = None,
                 debug: bool = False, llm_regions=None, engine_used: str = "opencv",
                 rect=None):
    """返回 (PipelineResult, debug_image|None, rectified_image)。

    rectified_image 即坐标基准图 (正畸后); app 导入后应显示它, 而非用户原图。
    llm_regions: 若提供(AI 粗框), 用它过滤 OpenCV 候选(精修)而非面积拐点压噪。
    rect: 若提供 (RectifyResult, 可来自 AI 找板正畸), 直接复用, 不再跑 OpenCV 正畸;
          这样 AI 粗框检测与本流水线共用同一张正畸基准图, 坐标对齐。
    """
    cfg = cfg or Config()
    warnings: List[str] = []

    # ① 正畸 (优先复用上游已算好的结果; 否则纯 OpenCV)
    rec = rect if rect is not None else R.rectify(bgr, cfg)
    img = rec.image
    if not rec.rectified:
        warnings.append(rec.note)
    h, w = img.shape[:2]

    # ② 光照归一化
    gray = P.to_gray_clahe(img, cfg)

    # ④ 基线 / 溶剂前沿 (先于③, 为③提供 ROI)
    ln = L.detect_lines(gray, cfg)
    if ln.baseline_y is None:
        warnings.append("未检到基线铅笔线, 待用户事后拖动给定")
    if ln.front_y is None:
        warnings.append("未检到溶剂前沿, 待用户事后拖动给定")

    # ③ 自动极性二值化 (hat 斑点检测, 用正畸后 BGR)
    bin_res = B.auto_binarize(img, ln.baseline_y, ln.front_y, cfg)
    if bin_res.uncertain:
        warnings.append("极性不确定 (明暗斑响应接近), 建议用户确认显色方式")

    # ⑤ 斑点候选 (形状无关) + ⑥ 泳道; 有 AI 粗框则用其精修
    sp = S.detect_spots(bin_res.binary, bin_res.gray, bin_res.polarity, bin_res.roi,
                        ln.baseline_y, ln.front_y, float(h * w), cfg,
                        keep_regions=llm_regions)

    spots_json = [{
        "x": round(s.x / w, 5), "y": round(s.y / h, 5),
        "bbox_norm": [round(s.bbox[0] / w, 5), round(s.bbox[1] / h, 5),
                      round(s.bbox[2] / w, 5), round(s.bbox[3] / h, 5)],
        "area_px": s.area, "lane": s.lane, "shape": s.shape,
        "rf": (round(s.rf, 4) if s.rf is not None else None),
    } for s in sp.spots]

    result = PipelineResult(
        width=w, height=h, rectified=rec.rectified,
        rectify_confidence=round(rec.confidence, 3),
        polarity=bin_res.polarity, polarity_uncertain=bin_res.uncertain,
        minority_frac=round(bin_res.minority_frac, 4),
        baseline_y_norm=(round(ln.baseline_y / h, 5) if ln.baseline_y is not None else None),
        front_y_norm=(round(ln.front_y / h, 5) if ln.front_y is not None else None),
        baseline_from=ln.baseline_from, front_from=ln.front_from,
        n_lanes=len(sp.lane_bounds), spots=spots_json, warnings=warnings,
        engine_used=engine_used,
    )

    # 显示用图: 在几何正畸图上叠加扫描件增强 (检测已在未增强图上完成, 坐标不变)
    display = E.enhance_scan(img, cfg) if cfg.enhance_enabled else img

    dbg = _draw_debug(display, ln, bin_res, sp) if debug else None
    return result, dbg, display


def _draw_debug(img, ln, bin_res, sp) -> np.ndarray:
    vis = img.copy()
    h, w = vis.shape[:2]
    x0, y0, x1, y1 = bin_res.roi
    cv2.rectangle(vis, (x0, y0), (x1, y1), (120, 120, 120), 1)  # ROI 灰框
    # 泳道分隔 (青)
    for (lo, hi) in sp.lane_bounds:
        cv2.line(vis, (hi, y0), (hi, y1), (200, 160, 0), 1)
    # 基线(绿) / 前沿(蓝)
    if ln.baseline_y is not None:
        yy = int(ln.baseline_y)
        cv2.line(vis, (0, yy), (w, yy), (0, 200, 0), 2)
        cv2.putText(vis, "baseline", (5, yy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
    if ln.front_y is not None:
        yy = int(ln.front_y)
        cv2.line(vis, (0, yy), (w, yy), (255, 80, 0), 2)
        cv2.putText(vis, "solvent front", (5, yy + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 80, 0), 2)
    # 斑点框 (红) + Rf
    for s in sp.spots:
        x, y, bw, bh = s.bbox
        cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
        if s.rf is not None:
            cv2.putText(vis, f"{s.rf:.2f}", (x, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return vis
