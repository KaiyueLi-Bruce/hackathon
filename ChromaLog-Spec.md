# ChromaLog — TLC 视觉识别与化学实验档案（macOS）

> 工作名 **ChromaLog**（备选名见附录 C，待定稿）。
> 本文档是交给 **Claude Code** 的构建规格：定义目标、技术栈、功能、数据模型、UI 方向、里程碑。
> 平台核心：**macOS 原生**。原则：**本地优先、注重 UI、手动优先 + 自动增强**。

---

## 1. 一句话定位

拍一张 TLC（薄层色谱）板照片，自动标定、计算 Rf 值、生成可解读的实验结论，并把结果沉淀为可搜索的本地化学实验档案。

---

## 2. 背景与痛点

TLC 是有机 / 药化 / 天然产物实验室每天要做几十次的基础操作，用来监控反应进度、判断纯度、比对化合物。但现状很原始：

- 在 UV 灯下肉眼看板，拿尺子手量斑点距离，手算 `Rf = 斑点迁移距离 / 溶剂前沿距离`。
- 结果要么画进纸质实验本，要么拍张手机照丢进相册——零散、不一致、无法检索。
- Rf 只有配上溶剂体系 / 板型 / 显色方式才有意义，而这些条件常常没被记下来，导致难以复现。

**价值主张（面向非化学评委也讲得通）**：把一件繁琐的手工测量工作自动化，并把记录变成可搜索的电子实验档案。"乱糟糟的照片 → 干净的数字化平板 + 自动 Rf + 可检索档案"这个转变，谁都看得懂。

---

## 3. 目标用户

有机 / 药物化学 / 天然产物方向的研究生、研究员、QC 人员；个人或小型实验室。

---

## 4. 30 秒 Demo 脚本（北极星，团队和 Claude Code 都对齐这条线）

1. 拖入一张真实 TLC 板照片（多泳道、UV 下拍摄）。
2. 拖动两条参考线：基线（原点）和溶剂前沿；点选几个斑点。
3. 右侧即时出现 Rf 表，中间生成一块标准化重绘的"数字平板"。
4. 点"AI 分析"，自动生成一份带结论的实验小报告（反应是否完成、产物在哪、下一步建议）。
5. 保存 → 进入档案；演示一次搜索（"找产物 Rf≈0.4 的实验"）。

**安全网**：全流程用手动点选就能 100% 跑通，自动识别只是锦上添花——现场绝不翻车。

---

## 5. 技术栈与架构（已定）

| 层 | 选型 | 说明 |
|---|---|---|
| 前端 / 外壳 | **SwiftUI**（macOS 14+） | 原生、漂亮、贴合 HIG；Claude Code 擅长生成 SwiftUI |
| 视觉处理 | **Python sidecar + OpenCV** | 作为本地子进程；MVP 阶段纯 SwiftUI/Core Graphics 手动点选即可，不依赖它 |
| Swift ↔ Python 通信 | **本地 HTTP（FastAPI）**，JSON 往返 | 简单、易调试；备选：标准输入输出 / 进程管道 |
| 本地存储 | **SQLite**（建议 GRDB.swift）+ 文件系统存原图与标注 | 本地优先，数据不出本机 |
| AI 分析 | **Anthropic API** | 生成结果分析与报告；用户填 API key，存 macOS Keychain |
| 打包 | Xcode app bundle | 黑客松阶段可先要求本机装 Python 环境；后续再用 PyInstaller / embedded venv 打进 bundle |

**架构示意**

```
┌─────────────────────────────────────────────┐
│            SwiftUI App (macOS)               │
│  Sidebar  │   Workspace 画布   │  Inspector  │
│  导航      │  图像+标注+Rf       │  结果/AI     │
└───────┬───────────────┬───────────────┬──────┘
        │ GRDB          │ 本地 HTTP      │ HTTPS
        ▼               ▼               ▼
   SQLite + 文件     Python sidecar    Anthropic
   (本地档案)        (OpenCV 斑点检测)   (分析/报告)
```

**为什么这么选**：SwiftUI 给最"Mac"的漂亮原生体验；OpenCV 留在 Python 生态最成熟、文档最全（开发者无需 CV 经验，由 Claude Code 生成）；手动 MVP 让 app 在 CV 接好之前就能完整跑通和演示。备选栈（Tauri + React + Python）见附录 A。

---

## 6. 信息架构与界面布局（画布优先 Canvas-first）

> **设计原则更新**：放弃通用的「三等分三栏」，改为**画布优先**——导航瘦成图标栏把空间让给板子，工具条浮在画布上，检视器做成上下文卡片栈。所有 app 内文案为**英文**；所有界面背景使用系统色,**跟随系统浅色 / 深色自动切换**。

- **左侧 Icon Rail（窄图标导航，~54px）**
  - 图标项：Experiments（实验树）、All Plates、Compounds（指纹库）、Search；底部 Settings
  - 选中项高亮（强调色填充）；hover 展开浮层显示项目/实验列表，不常驻占宽
  - 可完全折叠，进入「Focus / 纯画布标定」模式

- **中栏 Canvas（主舞台，占据最大空间）**
  - 当前板图像 + 标注画布：Baseline、Solvent front 两条可拖拽参考线，斑点标记
  - **浮动玻璃工具条**（悬浮在画布底部居中，非顶部整行）：Import · Spot · **Auto-detect**（高亮为主操作）· Redraw
  - **底部 Reaction time course 胶片条**：多块板按时间排成 filmstrip，当前板高亮边框，一眼看出原料消失 / 产物长出；末尾「＋」加板

- **右侧 Inspector（上下文卡片栈，~208px）**
  - 顶部 **Segmented control** 切换：Results / Conditions / AI（不是常驻三 Tab 面板）
  - **Results**：Rf values 卡（tabular 数字对齐）、泳道与斑点列表、Co-spot check 判定卡
  - **Conditions**：solvent system/ratio、stationary phase、visualization、plate type
  - **AI**：Generate AI report 按钮 + 报告预览 + 对板问答（chat）
  - 可隐藏，配合左栏折叠进入纯画布模式

---

## 7. 核心工作流（标定 → Rf）

1. 用户拖入 / 导入照片。
2. 拖动两条水平参考线：**基线**（原点 origin）与**溶剂前沿**（solvent front）。
3. 点选斑点（或自动检测后微调），每个斑点一个标记。
4. **Rf 计算**（图像坐标 y 向下，基线在下、前沿在上）：

   ```
   Rf = (baselineY − spotY) / (baselineY − frontY)
   ```

   结果应落在 0–1；越界则提示标定有误。若做了透视校正，先对坐标做透视变换再计算。
5. 生成**数字平板**：把原板标准化重绘成干净示意图（统一比例、标注 Rf）。
6. 保存为一条 Experiment 记录（含原图、标注、Rf、条件）。

---

## 8. 功能清单（按优先级，范围已锁定，不再新增）

### MVP（必做 · 黑客松核心 Demo）

- 图片导入 + 手动标定（基线 / 前沿 / 斑点）
- 自动 Rf 计算 + Rf 表
- 数字平板重绘
- 单板 AI 报告（目的 / 条件 / 观察 / 解读 / 下一步）
- 本地保存 + 列表浏览 + 基本搜索
- 漂亮的 macOS 原生 UI + 暗色模式

### 加分（有余力再做）

- OpenCV 自动斑点检测（手动兜底永远保留）
- 多泳道识别 + 标签（SM / 反应液 / 产物 / 共点 / 标准品）+ 共点判断
- 反应时程：多块板拼成时间线，展示原料消失 / 产物长出
- 条件元数据完整化 + 按条件检索
- 溶剂优化 / 上柱建议
- 报告导出 PDF / Word + Rf 数据导出 CSV
- 斑点强度估计（相对含量参考，注明非定量）

### 愿景（写进 pitch，不一定动手）

- 化合物 TLC 指纹库（多溶剂 Rf 匹配反查身份）
- 纸质实验本 OCR 数字化并入档案
- NMR / MS / IR / HPLC 统一分析档案
- 实验记录版本历史（复用 git 思路）
- 结构编辑器集成、跨档案分析仪表盘、移动端拍照采集

---

## 9. 数据模型

| 实体 | 关键字段 |
|---|---|
| Project | id, name, target（合成目标）, createdAt |
| Experiment | id, projectId, title, date, notes, conclusion, conditionsId, plates[] |
| Plate | id, experimentId, imagePath, baselineY, frontY, perspectiveTransform?, channel（UV254/UV365/stain）, capturedAt |
| Lane | id, plateId, label（SM/RXN/product/co-spot/standard）, xRange |
| Spot | id, plateId, laneId?, x, y, rf, intensity?, note |
| Conditions | id, solventSystem, ratio, stationaryPhase, visualization, plateType |
| Compound | id, name, structure?, fingerprints:[{ solventSystem, rf }] |
| Report | id, experimentId, markdown, model, createdAt |

---

## 10. AI 分析与报告规格

- **输入**：检测到的结构化数据（每块板的 Rf、泳道标签、条件、强度）+（可选）板图。
- **输出**：固定章节的结构化报告、反应结论判定、下一步建议、可对话追问。
- **报告章节**：目的 / 条件 / 观察（含 Rf 表）/ 解读 / 下一步 / 备注。
- **提示策略要点**：
  - 强调 TLC 为定性 / 半定量，结论给出置信度与依据。
  - **只基于传入的真实数据**，不臆造数值。
  - 共点、反应完成度等判断要说明推理（"产物斑点与标准品共点对齐 → 判定同一化合物"）。

---

## 11. UI/UX 设计方向（画布优先，重点：要"漂亮"）

**整体风格**：macOS HIG 原生质感；图标栏用半透明材质（vibrancy）；留白充足；SF Pro 字体；圆角克制；无多余阴影。**暗色模式作为一等公民——所有背景跟随系统浅色 / 深色自动切换**，不做独立主题。

**语言**：所有 app 内文案为**英文**（Import / Spot / Auto-detect / Redraw / Results / Conditions / AI / Rf values / Co-spot check / Generate AI report 等）。

**配色**：主强调色用实验室冷色（teal 或 indigo）；斑点用一组可区分的标记色——SM=indigo、Product=teal、By-product=coral（避免纯红绿，照顾色弱）；数字一律 tabular 对齐。

**布局范式（画布优先）**：窗口分三区但**非三等分**——左侧 ~54px 图标 rail、中间画布吃满主空间、右侧 ~208px 上下文检视器。左右两侧均可折叠 → 进入纯画布「Focus」标定模式。

**关键组件**：
- macOS 窗口 chrome（traffic-light + 居中标题 `Reaction A · Plate 3 · UV254` + 右侧 undo/share）
- 可拖拽、带吸附的 Baseline / Solvent front 标尺
- 可点选、可微调的斑点标记
- **悬浮玻璃工具条**（画布底部居中）：Import · Spot · Auto-detect（主操作高亮）· Redraw
- **Reaction time course 胶片条**（画布底部）：多板时间线缩略，当前板高亮边框，末尾「＋」加板
- 右栏 **Segmented control**（Results / Conditions / AI）+ 卡片栈：Rf values 卡、Co-spot check 判定卡、Generate AI report 按钮
- 数字平板重绘视图
- 报告预览（Markdown 渲染）
- 档案缩略图网格（带板图预览 + 关键元数据）

**交互细节**：拖拽导入图片；画布缩放 / 平移；标定线吸附；右栏分段切换；左右栏可隐藏；常用操作配键盘快捷键。

**三个关键界面（文字线框）**

1. 标定工作台（画布优先）
```
┌────────────────────────────────────────────────────────┐
│ ● ● ●        Reaction A · Plate 3 · UV254      ↶  ⤴      │  title bar
├──┬──────────────────────────────────────────┬──────────┤
│⚗ │            [ plate image ]                │ Results  │  segmented:
│▦ │        ── Solvent front ───────           │ ──────── │  Results|Cond|AI
│⊙ │         ●(SM) ●(Product) ●(Std)           │ Rf values│
│🔍│        ── Baseline ──────────              │ SM   0.62│
│  │                                           │ Prod 0.71│
│  │   ╭ Import · Spot · [Auto-detect] · Redraw ╮│ Co-spot ✓│
│  │   ╰───────── floating toolbar ───────────╯ │[AI report]│
│⚙ │  Reaction time course: ▯ ▯ [▮] ＋   0→2→4h │          │
└──┴──────────────────────────────────────────┴──────────┘
 icon rail            canvas (dominant)          inspector
```

2. 报告 / 结果检视：右栏 AI 段展示生成的报告 + "Export PDF" + 对话框追问。

3. 档案搜索网格：顶部搜索条（compound / project / Rf range / solvent / visualization），下方板图缩略图卡片网格。

---

## 12. 黑客松构建里程碑（建议顺序）

- **M0 骨架**：SwiftUI 画布优先布局（Icon rail + Canvas + Inspector，左右可折叠）+ 拖入并显示图片；英文 UI + 跟随系统明暗。
- **M1 核心算法**：手动标定（基线/前沿/斑点）+ Rf 计算 + Rf 表（纯 Swift，无需 OpenCV）。
- **M2 持久化**：数字平板重绘 + 本地 SQLite 保存 + 列表浏览。
- **M3 AI**：接 Anthropic API，生成单板报告。
- **M4 打磨**：美化 UI + 暗色模式 + 跑通 Demo 脚本。
- **M5 加分（全自动 OpenCV 流水线）**：Python sidecar + OpenCV 跑通**全自动流水线**——自动正畸（找板+透视校正，失败跳过）→ 光照归一化 → 自动极性二值化（少数派=斑点）→ Hough 识别铅笔基线/前沿 → 斑点候选 → 泳道归组；零训练即可演示，**手动只做事后拖动矫正**（详见附录 D.2）。
- **M5+ 模型升级（三模型，全 ONNX，逐个热替换）**：(a) 板分割 U-Net/DeepLab → 正畸来源 B；(b) sklearn `partial_fit` 增量 patch 分类，「越标越准」；(c) Ultralytics YOLO 高精度检测（数据攒够后最后上）。下游接口不变（详见附录 D.3 / D.8）。

> 顺序原则：先把可演示的主线（M0–M4）做完做稳，自动检测（M5）放最后；M5 全自动但事后手动矫正永远兜底，保证 Demo 稳定。模型（M5+）在 OpenCV 主线跑通后再逐级接入。

---

## 13. 与 Claude Code 对接说明

**建议仓库结构**
```
chromalog/
├── App/            # SwiftUI macOS 应用（Xcode 工程）
├── cv/             # Python sidecar：FastAPI + OpenCV
├── docs/           # 本规格文档及设计稿
└── samples/        # 真实 TLC 板照片（测试 + Demo）
```

**给 Claude Code 的第一步指令（示例）**
> "按本规格 M0：用 SwiftUI 搭一个 macOS **画布优先**窗口——左侧 ~54px 图标 rail、中间 Canvas 吃满主空间、右侧 ~208px Inspector（左右两栏均可折叠进入 Focus 模式）。Canvas 支持拖入图片并显示，底部预留浮动工具条与 Reaction time course 胶片条位置。**所有 UI 文案用英文，背景全部用系统色跟随明暗自动切换**，先不接 CV 和 AI。"

**需要你（Bruce）准备**
- 几张真实 TLC 板照片：含多泳道、UV 与显色各一张。
- Anthropic API key（用于 M3）。
- 本机 Python 环境（用于 M5 的 sidecar）。

**协作约定**：先做 MVP 主线；自动检测留到最后；手动点选永远作为兜底，保证 Demo 稳定。

---

## 14. 开放问题 / 待定

- 产品名定稿（见附录 C）。
- Swift ↔ Python 通信：FastAPI 本地端口 vs 进程管道。
- 存储选型：GRDB（SQLite）vs Core Data。
- 黑客松是否就打包 Python，还是要求本机环境。

---

## 附录 A — 备选技术栈（Tauri + React）

若更想用 Web 技术做 UI：Tauri（Rust 外壳）+ React 前端 + Python sidecar（或 OpenCV.js）。优点：UI 易做得漂亮、跨平台、Demo 只需开浏览器；缺点：不如 SwiftUI"原生 Mac"。鉴于"以 macOS 为核心 + 漂亮原生 UI"，正选仍是 SwiftUI。

## 附录 B — TLC 术语速查（给非化学队友 / 评委）

- **TLC**：薄层色谱，分离 / 鉴定化合物的常用实验。
- **Rf 值**：斑点迁移距离 ÷ 溶剂前沿距离，0–1，是该条件下化合物的"指纹"。
- **基线 / 原点**：点样起始线。**溶剂前沿**：展开剂爬升到的上沿。
- **泳道 lane**：一列样品。**共点 co-spot**：把两样品点在一起，判断是否同一化合物。
- **显色**：UV254/365 或 KMnO₄ / 茚三酮 / 碘缸等，让斑点可见。

## 附录 C — 备选产品名

ChromaLog（工作名）、RfLab、PlatePilot、Spotter、ChromaBook、PlateMate。

---

## 附录 D — 自动检测与正畸方案（全自动流水线 + 跨平台学习引擎）

> **路线决策：跨平台学习引擎。** 学习引擎做成与平台无关的独立 Python 组件，模型一律用 **ONNX** 做可移植格式，三平台（macOS / Windows / Linux）通吃。UI 外壳以 SwiftUI 做 Mac 精品版；Win/Linux 若要复用，只需另做壳子，**学习引擎与模型完全共用**。Apple Create ML / Core ML 因锁定单一平台，本项目不采用。
>
> **交互原则更新（v4）：全自动优先，手动只做事后矫正。** 对标现代扫描类 App——不要求用户手动拖四角、不要求用户手动画基线/前沿。整条流水线自动跑完出结果后，再允许用户拖动基线/前沿等做微调。手动操作从「输入步骤」降级为「事后安全网」。

### D.1 设计目标

把 M5 的「自动斑点检测」从纯固定规则，升级为**全自动正畸 + 检测**、并**随标注增长而变准**的系统，满足三个硬约束：**(1) 跨平台**；**(2) 最小数据即可见效**；**(3) 全自动，无需手动定位**。事后手动矫正永远保留，保证 Demo 稳定。

### D.2 全自动流水线（处理顺序已定）

```
原图
 → ① 自动找板 + 透视校正（失败则跳过，用原图继续，并标记低置信度）
 → ② 光照归一化（CLAHE，UV 辉光/不均匀光照必做）
 → ③ 二值化（自动极性：少数派=斑点）
 → ④ Hough 检测基线 / 溶剂前沿（铅笔线）
 → ⑤ 板内有效区斑点候选检测（连通域 + 面积/圆度/位置过滤）
 → ⑥ 按泳道归组（x 投影找峰切列）
 → 出结果（Rf 表 + 数字平板）
 → ⑦ 用户可拖动基线/前沿 → 实时重算 Rf（事后矫正，唯一手动入口）
```

**① 自动正畸（无手动四角）**：掩膜 → 最大轮廓 → `approxPolyDP` 拟合四角 → 单应矩阵 `warpPerspective` 拉平。**掩膜有两个来源、共用同一套「掩膜→四角→warp」后端**：
- 来源 A：**OpenCV** 边缘/阈值（Canny + 形态学），零训练，易图直接出掩膜，第一天可演示。
- 来源 B：**文档分割模型（小型 U-Net / DeepLabV3-MobileNet）→ ONNX**，难图（杂背景、贴边、低对比、UV 反光）兜底。
- 降级逻辑：先跑 A，四边形置信度低（面积/长宽比/角点不合理）→ 调 B → 仍失败 → **跳过校正用原图**，标记「低置信度」，绝不因一张难图整链崩溃。

**③ 自动极性二值化（少数派 = 斑点）**：先 Otsu/自适应阈值分两类，**取面积较小的一类为斑点（前景）**，自动适配 UV254（暗点亮底）/ UV365（亮点暗底）/ 显色剂（暗点亮底），无需用户预先选择显色方式。安全栏：
- 统计面积比前，**排除横贯全板的长细线**（基线/前沿）与板边/阴影；
- 只在**基线与前沿之间的有效区**内统计；
- 必须先做 ① 透视校正与 ② 光照归一化，否则阈值会被光照/形变干扰；
- 面积比接近 50/50（少数派假设失效，如重显色/大片拖尾）→ 判「不确定」，回退用户确认极性；
- `Conditions.visualization` 字段降级为**可选一致性校验**（用户填了就核对，不填也能跑）。

**④ 自动基线 / 溶剂前沿（纯识别铅笔线）**：**默认认定基线与溶剂前沿两条线均由用户用铅笔在板上画出**——这是本项目的标准操作约定。铅笔线特征为**细、暗、近水平、横贯板宽** → `HoughLinesP` 检测，按角度（近水平）、长度（占板宽大部分）、暗度过滤；正常筛出两条横线 → **下线=基线（原点）、上线=溶剂前沿**。这与 ③ 中「被排除的长横线」自洽复用，逻辑最干净。
- 兜底（仅当少于两条线时触发）：若只检到一条 → 视为基线，前沿回退检测**干湿分界的亮度/纹理突变**或取斑点群最高位置之上一点估计；一条都没检到 → 全交 ⑦ 用户拖动给定。默认路径不依赖这些兜底。

### D.3 模型清单（共 3 个，全部 ONNX，按需逐级上）

| # | 模型 | 职责 | 框架 | 何时上 |
|---|---|---|---|---|
| 1 | **板分割 U-Net / DeepLabV3-MobileNet** | 自动找板掩膜（正畸来源 B） | PyTorch 训 → `torch.onnx` 导出 | M5+ 升级（OpenCV 兜底先行） |
| 2 | **斑点 patch 二分类** | 判候选「是不是真斑点」，**在线增量**「越标越准」 | scikit-learn `SGDClassifier.partial_fit` → `skl2onnx` | M5 起步即可接 |
| 3 | **斑点检测 YOLO** | 高精度小目标检测（精度天花板） | **Ultralytics YOLO（底层 PyTorch，选路线 A 而非裸 PyTorch）** → `torch.onnx` | M5++ 数据攒够后 |

三者均经 ONNX 落地、**ONNX Runtime**（官方支持 Win/macOS/Linux，纯 CPU 可跑）推理。引擎以 **Python sidecar + FastAPI**（本地 HTTP，JSON）对接 SwiftUI，与第 5 节主架构一致，引擎本身平台无关。**模型边界要清楚**：模型 1 只管正畸，不碰基线/前沿（Hough 负责）与斑点；勿让模型清单膨胀。

### D.4 最小数据配方（你只有 3 张原图也能起步）

1. **候选**：OpenCV（或 SAM）零训练吐出疑似斑点 —— 一块板即产出几十个候选样本。
2. **合成预训练**：用 NumPy/Pillow/OpenCV 在真实空白板背景上程序化画高斯亮斑，生成上千张**带完美标注**的假板，预训练 patch 分类器 / 检测头；正畸模型 1 同理——把矩形板贴到各种背景做合成分割数据（掩膜天然完美）。
3. **零样本标注**：用 **SAM** 对真实照片零样本出掩膜/斑点，当作标签，省人工描边。
4. **真实微调**：用户手动修正的几十~上百个真实样本做 fine-tune（冻结主干、只训头部）。
5. **增量学习**：每次人工修正 → 存入 SQLite → `partial_fit` 在线更新，立即「越标越准」。
6. **主动学习**：模型把**最不确定**的候选优先丢给用户确认，同样标注量收敛最快。

数据增强（水平翻转、亮度/对比度抖动、加噪、模拟不同显色色调）全程免费扩样；**注意 Rf 依赖垂直几何，禁止剧烈竖向拉伸/旋转**（透视校正本身允许且必需）。

### D.5 标注闭环（「越标越准」的核心机制）

```
全自动流水线出结果 → 用户事后拖动基线/前沿、增删斑点(矫正)
        │ 结构化存入 SQLite(Spot 表)
        ▼
   训练集自动增长
        │ 攒够阈值 / 后台触发
        ▼
   sidecar 增量重训 → 导出新 .onnx
        │ 热替换
        ▼
   下次更准 → 主动学习只挑最不确定的问你
```

模型对不确定样本主动提问，用户矫正一条信息量最大；体验上即「它问你的越来越少」。

### D.6 架构示意（学习引擎解耦版）

```
┌─────────────────────────────────────────────┐
│            SwiftUI App (macOS 精品壳)         │   ← Win/Linux 另做壳，复用下方引擎
│  Sidebar  │   Workspace 画布   │  Inspector  │
└───────┬───────────────┬───────────────┬──────┘
        │ GRDB          │ 本地 HTTP      │ HTTPS
        ▼               ▼               ▼
   SQLite + 文件   跨平台学习引擎(Python)   Anthropic
   (本地档案/      正畸(OpenCV/U-Net) +      (分析/报告)
    训练集)        Hough 线 + 斑点检测       
                  (sklearn/YOLO) + 主动学习
                         │
                  模型文件 .onnx ×3 (三平台通用)
                         ▼
                  ONNX Runtime 推理
```

### D.7 开源参考实现（复用「架构 + 代码」，TLC 适配靠自有数据微调）

> 现成开源方案都是为**纸质文档/票据**做的，可直接复用「通用四边形正畸」那一层；但 TLC 板（UV 辉光、玻璃反光、与背景对比弱）域差异大，**预训练权重通常不能开箱即用**——复用其架构与后端代码，权重靠 SAM + 合成数据自行微调。

| 用途 | 开源项目 | 复用方式 |
|---|---|---|
| 正畸·几何起步（来源 A） | PyImageSearch / LearnOpenCV "Document Scanner" 教程代码、`jscanify`(OpenCV.js) | 直接抄「Canny→最大四边形→`getPerspectiveTransform`→warp」 |
| 正畸·模型升级（来源 B） | **LearnOpenCV "Automatic Document Scanner using DeepLabV3"**（代码+预训练权重）、U²-Net / DIS | 借架构与训练脚本，用自有数据微调出 ONNX |
| 零样本掩膜/标注 | **Segment Anything (SAM)** | 对真实图零样本出掩膜/斑点当标签 |
| 纯旋转纠偏（轻量） | `sbrunner/deskew`、`jdeskew` | 仅纠倾斜角，不纠透视；可作小角度兜底 |
| 曲面去畸（一般用不上） | `mzucker/page-dewarp` | 针对弯曲书页，平板 TLC 通常不需要 |
| 预训练数据集 | **SmartDoc-QA、MIDV-500 / MIDV-2020** | 手机拍文档+四角标注，供正畸模型预训练后迁移 |

> 注：以上项目与状态为编写时（约 2025 年中）的了解，落地前请再核实各仓库的最新版本与许可证。

### D.8 对里程碑的影响

第 12 节 **M5** 细化为**全自动 OpenCV 流水线**（正畸→光照归一化→自动极性二值化→Hough 基线/前沿→斑点候选→泳道归组），零训练即可演示，事后手动拖动矫正保留。**M5+** 拆出三模型升级，下游接口不变、逐个热替换：(a) 板分割 U-Net/DeepLab → 正畸来源 B；(b) sklearn 增量 patch 分类「越标越准」；(c) Ultralytics YOLO 高精度检测（最后上）。

---

*文档版本 v4 · 附录 D 升级为「全自动流水线 + 跨平台学习引擎」：正畸去掉手动四角（OpenCV→U-Net ONNX 降级）、二值化少数派自动极性、Hough 自动识别铅笔基线/前沿、手动降级为事后矫正；新增三模型清单（板分割/sklearn/YOLO 全走 ONNX）与开源参考实现小节；同步细化第 12 节 M5 / M5+。第 6/11 节为画布优先布局（英文 UI + 跟随系统明暗）。范围：以上功能为本阶段全部范围，不再扩展。*
