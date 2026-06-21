# ChromaLog CV sidecar — M5 全自动 OpenCV 流水线

跨平台 (Win / macOS / Linux)、纯 CPU、零训练的 TLC 板自动检测引擎。
对应 `ChromaLog-Spec.md` 附录 D.2。SwiftUI 通过本地 HTTP 调用，引擎本身平台无关。

## 流水线

```
原图
 → ① 自动找板 + 透视校正   (rectify.py;  失败→跳过用原图, 标记低置信度)
 → ② 光照归一化 CLAHE      (preprocess.py)
 → ④ Hough 铅笔基线/前沿   (lines.py;   在 gray 上先于③, 为③提供 ROI)
 → ③ 自动极性二值化        (binarize.py; 少数派=斑点, 自动适配 UV254/UV365/显色剂)
 → ⑤ 斑点候选 + ⑥ 泳道归组 (spots.py;   面积/圆度/位置过滤, x投影分泳道)
 → 归一化 JSON + 调试图     (pipeline.py)
```

掩膜来源可插拔：`rectify.rectify(bgr, cfg, mask_fn=...)`。默认 `plate_mask_opencv`（来源 A）；
M5+ 换成 U-Net/DeepLab → ONNX（来源 B），下游 “掩膜→四角→warp” 完全共用。

## 安装 & 运行

```bash
pip install -r requirements.txt          # 或仅 opencv-python-headless numpy 跑核心

# 本地 CLI（测试单图，导出调试可视化）
python run.py path/to/plate.jpg --debug out.png

# 启动 sidecar（SwiftUI 调用）
python -m chromalog_cv.server            # 127.0.0.1:8765
curl -F "file=@plate.jpg" "http://127.0.0.1:8765/detect?debug=true"
```

## SwiftUI 对接契约

`POST /rectify`（multipart 字段 `file`）→ `{width,height,rectified,rectify_confidence,note,engine_used,warnings,image_b64}`。
**只做正畸、不跑斑点检测**，供 app **导入即显示正畸图**（快）。
加 `?use_ai=true&or_model=openai/gpt-4o` + 头 `X-OpenRouter-Key: <key>` 即让 **AI 定位板主体**，
OpenCV 在该区域内**精修四角**再透视拉正；AI 失败绝不报错，自动回退纯 OpenCV（`warnings` 说明原因）。

`GET /config` → `{"tunable": {...默认值...}}`，供 Swift UI 初始化滑块。

`POST /detect`（multipart 字段 `file`，可选 `?debug=true`）→ JSON。
**三层检测**（spec 附录 D）：
1. **AI（在线）**：`POST /detect?use_ai=true&or_model=openai/gpt-4o` + 头 `X-OpenRouter-Key: <key>`。
   两处都交给 AI 粗定位、OpenCV 精修（`llm_detect.py`）：
   - **找板正畸**：AI 给板主体的粗略区域 → OpenCV 在区域内精修四角 → 透视拉正（`detect_plate` + `rectify.rectify_ai`）。
   - **斑点检测**：AI 给"真斑点的粗略区域" → OpenCV 在区域内精修出精确坐标/Rf（`detect_regions`）。
   正畸只算一次，既作 AI 斑点检测的输入又作流水线基准，坐标天然对齐。
2. **OpenCV（离线兜底）**：不传 `use_ai`、或没 key、或任一 AI 调用失败 → 该步自动回退纯 OpenCV，
   响应 `warnings` 会说明回退原因（找板与斑点两步独立降级，互不影响）。
3. **YOLO（终局）**：本地训练达标后接入（`Detector` 抽象预留，M5+）。

响应新增 `engine_used`：`opencv` | `ai+opencv` | `yolo`。AI 失败绝不报错，只降级。

**实时调参**：把调参旋钮作为 query 参数传入即可即时生效，无需改 Python，例如
`POST /detect?hat_thresh_k=2.5&knee_deviation=8`。可调项见下方“调参旋钮”表
（`hat_thresh_k`/`hat_kernel_frac`/`knee_deviation`/`knee_enabled`/`edge_margin_frac`/`roi_inset_frac`/`spot_max_area_frac`/`spot_min_area_frac`/`line_min_len_frac`）。响应字段：

| 字段 | 含义 |
|---|---|
| `image_b64` | **正畸后的图（PNG base64）= 坐标基准图。app 导入后应显示它，而非用户原图** |
| `width`,`height` | 校正图像素尺寸（坐标基准） |
| `rectified`,`rectify_confidence` | 是否做了透视校正及置信度 |
| `polarity`,`polarity_uncertain`,`minority_frac` | 极性判定；`uncertain=true` 时建议让用户确认显色方式 |
| `baseline_y_norm`,`front_y_norm` | 基线/前沿归一化 y（0..1，y 向下）；`null`=未检到，待用户拖动 |
| `baseline_from`,`front_from` | 来源：`hough`/`fallback`/`wetfront`/`none` |
| `n_lanes` | 泳道数 |
| `spots[]` | `{x,y, bbox_norm:[x,y,w,h], area_px, lane, shape, rf}`，坐标均归一化；`x,y` 为强度加权质心（拖尾更准）；`shape`∈`round/elliptical/tailing/irregular` |
| `warnings[]` | 降级/不确定提示，前端可气泡显示 |
| `debug_png_b64` | `?debug=true` 时返回的可视化 PNG（base64） |

**坐标全部归一化**，与分辨率无关。Rf = `(baselineY − spotY)/(baselineY − frontY)`，
**用户在前端拖动基线/前沿即可实时重算**，无需回调引擎（事后矫正，唯一手动入口）。

## 扫描件增强（CamScanner 风格）

借鉴 [ArashNasrEsfahani/Python-Document-Scanner-OpenCV](https://github.com/ArashNasrEsfahani/Python-Document-Scanner-OpenCV)
的增强阶段（`enhance.py`）：gamma 校正 → LAB 的 L 通道 CLAHE → 饱和度提升 → 反锐化掩膜，
让正畸后的板看起来像干净扫描件。

**关键**：增强**只作用于显示用的正畸图**（`image_b64`）；斑点检测与 Rf 仍跑在未增强的几何图上，
所以增强不会改变检测结果。开关与强度见 `config.py`（`enhance_enabled` / `enhance_gamma` /
`enhance_clahe_clip` / `enhance_saturation` / `enhance_sharpen`）。

正畸检测本身用多策略四点文档扫描（边缘 / Otsu / 角落背景色）+ "仅当板放在均匀背景上才裁切"的守门，
对已裁好的板不破坏，对桌面斜拍带背景的照片裁正成矩形。

## 调参旋钮（精度/召回平衡，在 `chromalog_cv/config.py`）

“尽量少标注”是默认取向：宁可漏掉淡而弥散的斑，也不把噪点当斑。需要更敏感/更保守时调：

| 参数 | 作用 | 调大 |
|---|---|---|
| `hat_thresh_k` (默认 4.0) | 二值化阈值 = 均值 + k·标准差 | 更保守，标更少（漏淡斑） |
| `hat_kernel_frac` (默认 0.18) | hat 核相对板短边，须 > 最大斑点 | 能抓更大的斑；过小则大斑只剩边缘被漏 |
| `knee_deviation` (默认 5.0) | 面积回归“严重偏离即停”的容忍度 | 标更多（越小越激进/标越少） |
| `edge_margin_frac` / `roi_inset_frac` | 排除板边/阴影/蓝底 | 排除更多边缘噪点（也可能切掉贴边的真斑） |
| `spot_max_area_frac` | 单斑面积上限 | 容许更大的过载斑 |

噪点抑制主要来自 hat 二值化（`hat_thresh_k`）；面积回归拐点（`knee_*`）是二级、按面积由大到小、严重偏离即停。

## 已知局限（M5 基线，均由后续升级或事后矫正覆盖）

- **溶剂前沿**：样张里常无可见前沿线时会取到顶边 → 用户拖动矫正；或确保按约定用铅笔画前沿。
- **非圆形斑点（椭圆 / 拖尾 / 不规则）已支持**：检测不靠圆度，只排除"线/边条"（跨度过大者）与触边连通域；斑点形状记录在 `shape` 字段，Rf 用强度加权质心。
- **照片内烤印的文字/标号**（如 PB、GLC1、泳道数字、手写 "TLC 1"）会被当作候选 → 真机照片一般无此标注，或事后删点；彻底解决靠 M5+ 学习层（按外观判真假，规则法做不到）。
- **连片/重叠斑点**可能合并为一个较大框 → M5+ 的 sklearn / YOLO 升级改善。
- **淡而弥散的低对比斑**会被保守阈值漏掉（"尽量少标注"的代价）→ 调 `hat_thresh_k` / `knee_deviation`，或事后手动补点。
- **90° 旋转板**（展开方向为左右、基线在侧边）：基线/前沿（水平 Hough）不适用，会退回整板 ROI 检斑点但 Rf 无意义 → 导入时先转正，或事后拖动基线。
- **多板并排 + 尺子同框**（如 TLC1.jpg）无法定位单板 → 降级用原图并告警；建议一次拍一块板。

## M5+ 升级位（接口不变，逐个热替换）

1. 板分割 U-Net/DeepLab → ONNX：替换 `rectify` 的 `mask_fn`（正畸来源 B）。
2. 斑点 patch 二分类 sklearn `partial_fit` → ONNX：在 `spots` 候选后加“真假斑点”过滤，闭环增量。
3. Ultralytics YOLO → ONNX：高精度检测，替换/增强 `spots`。
```
推理统一走 onnxruntime（见 requirements.txt 注释）。
```
