# 设计:斑点分类器在线增量学习 + 标注闭环(YOLO 留 seam)

- 日期:2026-06-20
- 状态:已批准,待转实现计划
- 关联:`ChromaLog-Spec.md` 附录 D.3(模型 2/3)、D.5(标注闭环)、D.6(架构)

## 1. 背景与目标

用户诉求:「先把 YOLO 接进来,然后能通过手动标定不断学习」。

经核对 spec 与现状,确认一个核心矛盾并据此调整方向:

- spec 中「通过手动标定不断学习/越标越准」的**真正机制是模型 2:sklearn `SGDClassifier.partial_fit`
  在线增量 patch 分类器**——小数据即起效,标一条立刻更准(定位「M5 起步即可接」)。
- **YOLO 是模型 3**,明确是「数据攒够后最后上」的精度天花板,走**离线 fine-tune→ONNX**,
  **本质上不做在线增量**。现状无模型、无数据(3–7 张图)、venv 未装 ML 依赖。

**结论(用户已确认)**:本轮做 sklearn 在线增量这条轨道——标注闭环 + patch 二分类「越标越准」,
并为 YOLO 留下清晰的离线热替换接口(seam);**本轮不接 YOLO 推理/训练**。

### 目标
1. 用户手动矫正斑点后,引擎能从矫正中学习,**下次检测对同类板更准**(抑制板角/反光/背景等误检)。
2. 学习闭环平台无关(Python sidecar),app 改动最小。
3. 留好 YOLO 热替换边界,下游(泳道/Rf/渲染)接口不变。

### 非目标(本轮明确不做)
YOLO 实际推理/训练;板分割 U-Net(正畸来源 B);ONNX 导出;主动学习 UI;显式「确认/否决」交互。

## 2. 总体数据流

```
检测: OpenCV 候选 ──► [若已训练] sklearn 打分(P≥阈值保留)  ──► 斑点
                      [未训练]   退回 AI 框过滤 / 面积拐点(现状)
                          ▲
保存矫正: app POST /learn 「正畸图 + 最终斑点 + 原始自动候选」
                          │ sidecar 派生样本:
                          │   保留的候选 / 用户新增 = 正样本
                          │   被删的候选          = 硬负样本 (板角/反光… 信息量最大)
                          │   随机背景            = 易负样本
                          ▼
                  SGDClassifier.partial_fit → 落盘 pkl + 累积样本 npz
                  → 下次检测立即更准 (越标越准)
```

## 3. Sidecar(Python)新增

### 3.1 `chromalog_cv/learn.py`
- **特征提取** `patch_features(bgr, bbox_norm) -> np.ndarray`:按归一化 bbox 从正畸图裁 patch,
  resize 到定长(如 24×24)灰度并标准化,展平为像素向量 + 少量手工统计(均值/对比度/Sobel 梯度能量)。
  定长输出,保证 `partial_fit` 维度稳定。
- **`SpotClassifier`**:封装 `StandardScaler`(用 `partial_fit` 在线更新均值方差)+
  `SGDClassifier(loss="log_loss")`(输出概率,支持 `partial_fit(classes=[0,1])`);
  方法 `update(X, y)`、`proba(X) -> p_real`、`save(path)`、`load(path)`、`is_trained`。
- **样本派生** `derive_samples(bgr, final_spots, auto_candidates) -> (X, y)`:
  - 候选 ↔ 最终斑点 用质心/IoU 匹配:被保留的候选 → 正;被删的候选(未匹配上)→ 硬负。
  - 用户新增(最终斑点里匹配不到任何候选的)→ 正。
  - 在板内随机采若干不与任何最终斑点重叠的位置 → 易负(数量与正样本数挂钩,平衡类别)。
- **持久化**:`cv/models/spot_clf.pkl`(分类器+scaler)、`cv/models/spot_samples.npz`
  (累积 X,y,供将来全量重训 / ONNX 导出)。

### 3.2 端点
- **`POST /learn`**:multipart 接收正畸图 + JSON `{final_spots:[bbox_norm…], auto_candidates:[bbox_norm…]}`;
  `derive_samples` → `SpotClassifier.update`(partial_fit)→ 落盘;
  返回 `{trained_total:int, batch:{pos:int, neg:int}, ok:true}`。任何异常返回 `{ok:false, error}`,不抛 500。
- **`GET /model`**:返回 `{trained:bool, n_samples:int, updated_at:str|null}`,供 app 显示学习状态。

## 4. 检测集成与优先级(三级热替换)

`pipeline.run_pipeline` 加载 `SpotClassifier`(若存在),把一个打分器传入 `spots.detect_spots`;
在现有候选(形状/线/边缘过滤后)之后增加一个过滤阶段,优先级:

1. **已训练分类器** → 对每个候选 patch 打分,`P(real) ≥ 0.5`(可配 `spot_clf_thresh`)保留 —— **替代面积拐点**。
2. **否则 AI 框**(若开 AI)→ 现有 `_filter_by_regions` + Plan C 兜底(已实现)。
3. **否则** → `_area_knee_cut` 面积拐点压噪(现状)。

`PipelineResult` 扩展:`engine_used` 增加 `+skl` 形态(如 `opencv+skl`、`ai+opencv+skl`);
新增布尔字段 `learned`(本次检测是否用了学习模型)。

## 5. App(Swift)改动(最小)

- **`AppStore`**:检测后把响应里的斑点另存为 `autoCandidates: [Spot]`(用户编辑前的原始候选,只读快照)。
- **`CVClient`**:新增 `learn(rectified:Data, finalSpots:[…], autoCandidates:[…]) async`(POST `/learn`)、
  `modelInfo() async -> {trained, nSamples}`(GET `/model`)。
- **`saveCurrentPlate()`** 成功后,调用 `learn(...)`(失败仅记日志,不阻断保存)。
- `/detect` 调用**始终隐含使用学习模型**(sidecar 有模型就用,无则退回现状)——**不新增 AI/学习开关**。
- **Inspector** 加一行状态:`Learned from N corrections`(启动/保存后刷新 `/model`)。

正畸图来源:`/detect` 响应的 `image_b64` 即正畸基准图,最终斑点与原始候选均为其归一化坐标,
直接回传给 `/learn`,坐标天然对齐。

## 6. 存储 / 模型生命周期 / YOLO seam

- 模型与样本存 `cv/models/`,**加入 `.gitignore`**(随用户数据增长,不入库)。
- **刻意偏离 spec**:先用 **pickle 在 sidecar 内直接跑,不导 ONNX**。理由:sidecar 本身就是 Python 运行时,
  pkl 最简且正确;ONNX 是为 Win/Linux 跨平台部署,留后续。`spot_samples.npz` 保留原始样本,
  将来可一键全量重训 + `skl2onnx` 导出,不丢数据。
- **YOLO seam**:在引擎层定义清晰边界——OpenCV 做「候选提议器(proposer)」、sklearn 做「打分器(scorer)」。
  将来 YOLO 作为「提议器+打分器」整体热替换这一层,下游(泳道/Rf/渲染)接口与 `PipelineResult` 不变。
  本轮只把函数边界与 `engine_used` 形态留好。

## 7. 依赖

`cv/requirements.txt` 取消注释并安装:`scikit-learn`(`SGDClassifier`/`StandardScaler`)。
**不装** `onnxruntime`/`skl2onnx`/`ultralytics`/`torch`(本轮非目标)。

## 8. 测试

- 单元:`patch_features` 维度稳定;`derive_samples` 在给定候选/最终斑点下正确分出正/硬负/易负数量。
- 端到端脚本:用合成板(`_debug/synthetic_*`)+ 训练图,模拟「检测→删掉板角误检→/learn→再检测」,
  断言学习后该类误检被抑制(P<阈值被滤掉),真斑点保留。
- 回归:无模型时 `detect` 行为与现状一致(走 AI 框 / 面积拐点)。

## 9. 风险

- 小数据下 SGD 易过拟合 → 用易负平衡 + 标准化 + 较保守阈值;`npz` 累积便于后续重训。
- 类别极不平衡(正样本少)→ 易负采样数与正样本挂钩,必要时 `class_weight`/采样平衡。
- patch 坐标对齐依赖「app 回传的正是 /detect 那张正畸图」→ 设计上强制用 `image_b64`,不另存。
