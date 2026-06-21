"""流水线可调参数。全部相对量 (相对板尺寸), 以便对任意分辨率稳定。"""

from dataclasses import dataclass, fields
from typing import Any, Dict


@dataclass
class Config:
    # ---- ① 正畸 ----
    work_max_side: int = 1400          # 处理前把长边缩到该值以内 (加速; 结果坐标归一化, 不受影响)
    plate_min_area_frac: float = 0.05  # 候选四边形面积须 >= 整图该比例 (板可能在大背景里占比较小)
    plate_quad_eps_frac: float = 0.02  # approxPolyDP 精度 (相对周长)

    # ---- ② 光照归一化 ----
    clahe_clip: float = 2.0
    clahe_grid: int = 8

    # ---- ④ Hough 铅笔线 ----
    line_min_len_frac: float = 0.45    # 线长须 >= 板宽该比例
    line_max_angle_deg: float = 6.0    # 与水平夹角上限
    line_merge_frac: float = 0.03      # y 距小于板高该比例的线合并为一条
    line_min_sep_frac: float = 0.25    # 基线/前沿须相隔板高该比例以上, 否则视为无效(防边缘双线塌缩)

    # ---- ③ 自动极性二值化 (top-hat/black-hat 斑点检测) ----
    roi_inset_frac: float = 0.05       # 有效区相对板边内缩 (排除板鳞/阴影)
    hat_kernel_frac: float = 0.18      # 形态学 hat 核相对板短边 (须略大于最大斑点, 否则大斑点只剩边缘)
    hat_thresh_k: float = 4.0          # 阈值 = 均值 + k*标准差 (越大越保守, 尽量少标注)
    polarity_uncertain_lo: float = 0.40  # Otsu 少数派占比落在 [lo,hi] -> 判"不确定"
    polarity_uncertain_hi: float = 0.60
    morph_frac: float = 0.006          # 形态学清理核相对板高

    # ---- ⑤ 斑点候选 (形状无关: 不靠圆度判斑点, 只排除"线") ----
    spot_min_area_frac: float = 8e-5   # 斑点面积相对板面积下限 (去噪点)
    spot_max_area_frac: float = 0.05   # 上限 (去污渍/大块)
    # "线"判据 (基线/前沿/划痕): 横贯板宽 或 纵贯板高 且 很细
    line_span_frac: float = 0.55       # 跨度 >= 板宽/板高该比例
    line_thin_frac: float = 0.025      # 且细边 <= 板高该比例 -> 判为线, 剔除
    spot_min_extent: float = 0.12      # 极低填充度才剔 (多为碎屑/空心环噪声), 椭圆/拖尾均保留
    tailing_elong: float = 2.0         # 纵向拉长 >= 此值标记为拖尾(仍是斑点)
    edge_margin_frac: float = 0.035    # 质心距图像边缘 < 该比例 -> 板边/阴影, 剔除

    # ---- ⑤b 面积回归拐点压噪 (尽量少标注: 由大到小, 严重偏离即停) ----
    knee_enabled: bool = True
    knee_min_points: int = 3           # 至少保留前 N 个才开始判拐点
    knee_deviation: float = 5.0        # "严重偏离": 实际面积 < 回归预测/该倍数 -> 停止 (越小越激进/标越少)
    knee_min_keep: int = 1             # 至少保留 1 个

    # ---- 扫描件增强 (借鉴 ArashNasrEsfahani/Python-Document-Scanner-OpenCV) ----
    # 仅作用于"显示用"正畸图; 斑点检测仍用未增强几何图, 不影响 Rf。
    enhance_enabled: bool = True
    enhance_gamma: float = 1.15        # >1 提亮中间调
    enhance_clahe_clip: float = 2.0    # LAB-L 通道局部对比 (CLAHE)
    enhance_saturation: float = 1.12   # 色彩饱和度
    enhance_sharpen: float = 0.6       # 反锐化掩膜强度 (0=不锐化)

    # ---- ⑥ 泳道归组 ----
    lane_smooth_frac: float = 0.03     # x 直方图平滑窗口相对板宽
    lane_min_gap_frac: float = 0.05    # 相邻泳道峰最小间距相对板宽

    # 允许经 sidecar 实时覆盖的旋钮 (Swift 端调参用)
    TUNABLE = ("hat_thresh_k", "hat_kernel_frac", "knee_deviation", "knee_enabled",
               "edge_margin_frac", "roi_inset_frac", "spot_max_area_frac",
               "spot_min_area_frac", "line_min_len_frac")

    @classmethod
    def from_overrides(cls, overrides: Dict[str, Any]) -> "Config":
        """仅用非 None 且属于 TUNABLE 的项覆盖默认值, 其余忽略 (防注入)。"""
        valid = {f.name for f in fields(cls)}
        cfg = cls()
        for k, v in overrides.items():
            if v is not None and k in cls.TUNABLE and k in valid:
                setattr(cfg, k, v)
        return cfg
