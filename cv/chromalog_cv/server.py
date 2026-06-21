"""跨平台 Python sidecar (FastAPI, 本地 HTTP, JSON)。

SwiftUI 通过本地端口调用 /detect, 引擎本身平台无关 (附录 D.2 / 第 5 节)。
启动:  python -m chromalog_cv.server   (默认 127.0.0.1:8765)
或:    uvicorn chromalog_cv.server:app --host 127.0.0.1 --port 8765
"""
from __future__ import annotations

import base64
import json
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Query, Header, Form
from fastapi.responses import JSONResponse

from .config import Config
from .pipeline import run_pipeline
from . import rectify as R
from .rectify import rectify as cv_rectify
from .enhance import enhance_scan
from . import llm_detect as L
from . import report as RPT
from . import learn as LN

app = FastAPI(title="ChromaLog CV sidecar", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok", "engine": "opencv-auto-pipeline", "version": "0.1.0"}


@app.get("/config")
def get_config():
    """返回可实时调节的旋钮及其默认值, 供 Swift UI 初始化滑块。"""
    d = Config()
    return {"tunable": {k: getattr(d, k) for k in Config.TUNABLE}}


@app.post("/rectify")
async def rectify_only(
    file: UploadFile = File(...),
    # ---- AI 找板正畸 (AI 粗定位 + OpenCV 精修四角); 失败自动回退纯 OpenCV ----
    use_ai: bool = Query(False, description="启用 OpenRouter AI 找板 + OpenCV 精修四角正畸"),
    or_model: Optional[str] = Query(None, description="OpenRouter 模型 id (视觉)"),
    x_openrouter_key: Optional[str] = Header(None, description="OpenRouter API key"),
):
    """只做正畸 (快): 导入时调用, 让画布立即显示正畸后的图。不跑斑点检测。

    use_ai+key+model 时让 AI 定位板主体, OpenCV 在该区域内精修四角再拉正; AI 失败绝不报错,
    自动回退纯 OpenCV 正畸 (warnings 说明原因)。
    """
    try:
        img = _decode(await file.read())
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    cfg = Config()
    rec, engine, warns = _rectify(img, cfg, use_ai, or_model, x_openrouter_key)
    disp = enhance_scan(rec.image, cfg) if cfg.enhance_enabled else rec.image
    payload = {
        "width": int(disp.shape[1]), "height": int(disp.shape[0]),
        "rectified": rec.rectified, "rectify_confidence": round(rec.confidence, 3),
        "note": rec.note, "engine_used": engine, "warnings": warns,
    }
    ok, enc = cv2.imencode(".png", disp)
    if ok:
        payload["image_b64"] = base64.b64encode(enc.tobytes()).decode("ascii")
    return payload


def _decode(buf: bytes) -> np.ndarray:
    arr = np.frombuffer(buf, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("无法解码图像")
    return img


def _rectify(img: np.ndarray, cfg: Config, use_ai: bool,
             or_model: Optional[str], key: Optional[str]):
    """统一正畸入口: OpenCV 优先, AI 仅在 OpenCV 失败/低置信时介入纠偏。
    返回 (RectifyResult, engine_used, warnings)。AI 失败绝不报错, 只降级。

    设计意义 (见与 detect_regions 同款的"AI 补 OpenCV 短板"哲学):
      纯 OpenCV 对"板坐落在大片均匀背景上 + 透视"这类图已能可靠拉正; AI 预裁切反而会
      破坏 OpenCV 找板所依赖的全图背景上下文, 把流程逼入更差的降级分支 (TLC_real_2 即此)。
      故先跑 OpenCV, 够可信就采信; 仅当其失败/低置信 (背景杂乱/板非最大均匀区) 才请 AI 定位纠偏。"""
    warns = []
    # ① 先跑纯 OpenCV (便宜, 且对干净背景的图已足够好)
    rec_cv = cv_rectify(img, cfg)
    if rec_cv.rectified and rec_cv.confidence >= cfg.rectify_cv_trust:
        return rec_cv, "opencv", warns

    # ② OpenCV 失败/低置信 -> 让 AI 找板纠偏 (AI 粗定位 + OpenCV 在该区域内精修四角)
    if use_ai and key and or_model:
        try:
            bbox, quad = L.detect_plate(img, key, or_model)
            rec_ai = R.rectify_ai(img, cfg, bbox=bbox, quad=quad)
            if rec_ai.rectified:
                return rec_ai, "ai+opencv", warns
            warns.append("AI 正畸亦未成功, 用 OpenCV 结果")
        except L.LLMError as e:
            warns.append(f"AI 找板不可用, 用 OpenCV 正畸: {e}")
        except Exception as e:
            warns.append(f"AI 找板异常, 用 OpenCV 正畸: {e}")
    return rec_cv, "opencv", warns


@app.post("/learn")
async def learn_endpoint(
    file: UploadFile = File(...),
    payload: str = Form(...),
):
    """从一次手动矫正在线增量训练斑点分类器 (设计 §3.2)。
    payload: {"final_spots": [[x,y]...], "auto_candidates": [[x,y]...]} 归一化质心。
    任何坏输入返回 ok:false, 不抛 500。"""
    try:
        img = _decode(await file.read())
        data = json.loads(payload)
        final_pts = [(float(p[0]), float(p[1])) for p in data.get("final_spots", [])]
        auto_pts = [(float(p[0]), float(p[1])) for p in data.get("auto_candidates", [])]
        baseline_y = data.get("baseline_y")
        front_y = data.get("front_y")
        baseline_y = float(baseline_y) if baseline_y is not None else None
        front_y = float(front_y) if front_y is not None else None
    except Exception as e:
        return JSONResponse(status_code=200, content={"ok": False, "error": str(e)})
    try:
        return LN.apply_correction(img, final_pts, auto_pts, Config(),
                                   baseline_y=baseline_y, front_y=front_y)
    except Exception as e:
        return JSONResponse(status_code=200, content={"ok": False, "error": str(e)})


@app.get("/model")
def model_endpoint():
    return LN.model_info(LN.CLF_PATH, LN.SAMPLES_PATH)


@app.post("/report")
async def report_endpoint(
    payload: str = Form(...),
    mode: str = Query("report", description="questions | report"),
    model: str = Query(...),
    x_openrouter_key: str = Header(None),
):
    """AI 实验报告 (spec §10)。
    payload: {"data": {...Rf/条件/时程...}, "notebook": "", "answers": ""}
    mode=questions -> {questions:[...]}; mode=report -> {markdown:"..."}。"""
    try:
        body = json.loads(payload)
        data = body.get("data", {})
        notebook = str(body.get("notebook", "") or "")
        answers = str(body.get("answers", "") or "")
    except Exception as e:
        return JSONResponse(status_code=200, content={"ok": False, "error": str(e)})
    try:
        if mode == "questions":
            return {"ok": True, "questions": RPT.generate_questions(data, x_openrouter_key, model)}
        return {"ok": True, "markdown": RPT.generate_report(data, notebook, answers,
                                                            x_openrouter_key, model)}
    except RPT.LLMError as e:
        return JSONResponse(status_code=200, content={"ok": False, "error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=200, content={"ok": False, "error": str(e)})


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    debug: bool = Query(False, description="返回 base64 调试图"),
    # ---- AI 粗框 (三层: AI->OpenCV) ----
    use_ai: bool = Query(False, description="启用 OpenRouter AI 粗框 + OpenCV 精修"),
    or_model: Optional[str] = Query(None, description="OpenRouter 模型 id (视觉)"),
    x_openrouter_key: Optional[str] = Header(None, description="OpenRouter API key"),
    # ---- 可选实时调参 (不传则用默认) ----
    hat_thresh_k: Optional[float] = Query(None, description="二值化阈值=均值+k·标准差, 越大越保守/标越少"),
    hat_kernel_frac: Optional[float] = Query(None, description="hat 核相对板短边, 须>最大斑点"),
    knee_deviation: Optional[float] = Query(None, description="面积回归严重偏离容忍度, 越小越激进/标越少"),
    knee_enabled: Optional[bool] = Query(None, description="是否启用面积回归拐点压噪"),
    edge_margin_frac: Optional[float] = Query(None, description="质心距边缘<该比例则剔(板边/阴影)"),
    roi_inset_frac: Optional[float] = Query(None, description="有效区相对板边内缩"),
    spot_max_area_frac: Optional[float] = Query(None, description="单斑面积上限(相对板面积)"),
    spot_min_area_frac: Optional[float] = Query(None, description="单斑面积下限(去噪点)"),
    line_min_len_frac: Optional[float] = Query(None, description="基线/前沿线长须>板宽该比例"),
):
    try:
        img = _decode(await file.read())
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    cfg = Config.from_overrides({
        "hat_thresh_k": hat_thresh_k, "hat_kernel_frac": hat_kernel_frac,
        "knee_deviation": knee_deviation, "knee_enabled": knee_enabled,
        "edge_margin_frac": edge_margin_frac, "roi_inset_frac": roi_inset_frac,
        "spot_max_area_frac": spot_max_area_frac, "spot_min_area_frac": spot_min_area_frac,
        "line_min_len_frac": line_min_len_frac,
    })
    # ① 正畸: AI 找板(粗) + OpenCV 精修四角; 失败/未启用回退纯 OpenCV。
    #    正畸只算一次, 既作 AI 粗框检测的输入又作流水线基准, 坐标天然对齐。
    rec, engine, extra_warn = _rectify(img, cfg, use_ai, or_model, x_openrouter_key)

    # ② AI 斑点粗框 (在同一张正畸图上); 失败/未启用 -> OpenCV 兜底
    llm_regions = None
    if use_ai and x_openrouter_key and or_model:
        try:
            reg = L.detect_regions(rec.image, x_openrouter_key, or_model)
            llm_regions = reg.regions
            engine = "ai+opencv"
        except L.LLMError as e:
            extra_warn.append(f"AI 斑点检测不可用, 回退 OpenCV: {e}")
        except Exception as e:
            extra_warn.append(f"AI 斑点检测异常, 回退 OpenCV: {e}")

    result, dbg, rect_img = run_pipeline(img, cfg, debug=debug,
                                         llm_regions=llm_regions, engine_used=engine,
                                         rect=rec)
    result.warnings = extra_warn + result.warnings
    payload = result.to_json()
    # 正畸后的图 (坐标基准): app 导入后显示它, 不显示原图
    ok, enc = cv2.imencode(".png", rect_img)
    if ok:
        payload["image_b64"] = base64.b64encode(enc.tobytes()).decode("ascii")
    if debug and dbg is not None:
        ok, enc = cv2.imencode(".png", dbg)
        if ok:
            payload["debug_png_b64"] = base64.b64encode(enc.tobytes()).decode("ascii")
    return payload


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765)
