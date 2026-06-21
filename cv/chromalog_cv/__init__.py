"""ChromaLog 跨平台 CV 引擎 (M5 全自动 OpenCV 流水线)。

设计原则 (见 ChromaLog-Spec.md 附录 D):
  - 全自动: 无需手动定位四角 / 基线 / 前沿; 手动只做事后矫正。
  - 跨平台: 纯 OpenCV + NumPy, 平台无关; 模型升级走 ONNX。
  - 优雅降级: 任一步失败均回退, 绝不让单张难图整链崩溃。

流水线顺序 (附录 D.2):
  ① 自动找板 + 透视校正  -> rectify.py
  ② 光照归一化 (CLAHE)   -> preprocess.py
  ④ Hough 铅笔基线/前沿  -> lines.py   (在 gray 上先于③, 为③提供 ROI)
  ③ 自动极性二值化       -> binarize.py
  ⑤ 斑点候选 + ⑥ 泳道归组 -> spots.py
  编排                   -> pipeline.py
"""

from .config import Config
from .pipeline import run_pipeline, PipelineResult

__all__ = ["Config", "run_pipeline", "PipelineResult"]
__version__ = "0.1.0"
