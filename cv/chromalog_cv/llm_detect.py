"""AI 粗框检测 (OpenRouter 视觉模型)。

设计 (见 ChromaLog-Spec.md 附录 D, "AI 粗框 + OpenCV 精修"):
  LLM **不**负责精确坐标/Rf —— 它只回答"哪些区域是真斑点"(粗略归一化框) +
  泳道数 + 显色类型。精确几何由 OpenCV 在这些区域内精修 (见 pipeline.py)。

这样彻底规避 LLM 坐标不准/臆造数值的问题, 几何精度全部来自 OpenCV。

调用方提供 OpenRouter api_key 与 model (app 端从 Keychain 取后随请求传入)。
任何失败都抛 LLMError, 由编排层回退到纯 OpenCV。
"""
from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np
import urllib.request
import urllib.error

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

PROMPT = (
    "You are assisting a TLC (thin-layer chromatography) plate analyzer. "
    "Look at this rectified TLC plate image. Return ONLY compact JSON, no prose.\n"
    "Identify the REAL sample spots (ignore pencil marks, lane labels, text, dust, "
    "glare, plate edges). Coarse bounding boxes are fine — exact pixels are not needed.\n"
    "JSON schema:\n"
    "{\n"
    '  "visualization": "UV254|UV365|stain|unknown",\n'
    '  "n_lanes": <int>,\n'
    '  "regions": [ {"x":<0..1>,"y":<0..1>,"w":<0..1>,"h":<0..1>,"lane":<int>} ]\n'
    "}\n"
    "Coordinates are normalized to the image (origin top-left, y down). "
    "Each region should loosely enclose one real spot."
)


class LLMError(Exception):
    pass


@dataclass
class LLMRegions:
    visualization: str = "unknown"
    n_lanes: int = 0
    regions: List[dict] = field(default_factory=list)   # 每项 {x,y,w,h,lane} 归一化


def _encode_png_b64(bgr: np.ndarray, max_side: int = 1024) -> str:
    h, w = bgr.shape[:2]
    scale = min(1.0, max_side / float(max(h, w)))
    img = cv2.resize(bgr, None, fx=scale, fy=scale) if scale < 1.0 else bgr
    ok, enc = cv2.imencode(".png", img)
    if not ok:
        raise LLMError("无法编码图像")
    return base64.b64encode(enc.tobytes()).decode("ascii")


def _extract_json(text: str) -> dict:
    # 容忍模型在 JSON 外多写文字: 抓第一个 {...}
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise LLMError("LLM 未返回 JSON")
        return json.loads(m.group(0))


PLATE_PROMPT = (
    "You are assisting a TLC plate scanner (like a document scanner). "
    "Find the TLC plate (the main rectangular subject) in this photo. "
    "Return ONLY compact JSON, no prose:\n"
    "{\n"
    '  "plate": {"x":<0..1>,"y":<0..1>,"w":<0..1>,"h":<0..1>},\n'
    '  "quad": [[x,y],[x,y],[x,y],[x,y]]\n'
    "}\n"
    "`plate` is the axis-aligned bounding box of the plate; `quad` (optional) is its "
    "four corners (top-left, top-right, bottom-right, bottom-left). "
    "All coordinates normalized to the image (origin top-left, y down)."
)


def _chat_vision(bgr: np.ndarray, api_key: str, model: str, prompt: str,
                 timeout: float) -> dict:
    if not api_key:
        raise LLMError("缺少 OpenRouter API key")
    b64 = _encode_png_b64(bgr)
    body = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }],
        "temperature": 0,
    }
    req = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://chromalog.local",
            "X-Title": "ChromaLog",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise LLMError(f"OpenRouter HTTP {e.code}: {e.read().decode('utf-8', 'ignore')[:200]}")
    except Exception as e:
        raise LLMError(f"OpenRouter 请求失败: {e}")
    try:
        content = payload["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content)
    except Exception:
        raise LLMError("OpenRouter 响应结构异常")
    return _extract_json(content)


def detect_plate(bgr: np.ndarray, api_key: str, model: str, timeout: float = 45.0):
    """让 AI 定位板(主体)。返回 (bbox dict|None, quad ndarray(4,2)|None), 归一化坐标。
    AI 只给粗略位置, 精确四角由 OpenCV 在该区域内精修。"""
    data = _chat_vision(bgr, api_key, model, PLATE_PROMPT, timeout)
    bbox = None
    p = data.get("plate")
    if isinstance(p, dict):
        try:
            bbox = {"x": float(p["x"]), "y": float(p["y"]),
                    "w": float(p["w"]), "h": float(p["h"])}
        except Exception:
            bbox = None
    quad = None
    q = data.get("quad")
    if isinstance(q, list) and len(q) == 4:
        try:
            quad = np.array([[float(pt[0]), float(pt[1])] for pt in q], dtype=np.float32)
        except Exception:
            quad = None
    if bbox is None and quad is None:
        raise LLMError("AI 未返回板位置")
    return bbox, quad


def detect_regions(bgr: np.ndarray, api_key: str, model: str,
                   timeout: float = 45.0) -> LLMRegions:
    data = _chat_vision(bgr, api_key, model, PROMPT, timeout)
    regions = []
    for r in data.get("regions", []):
        try:
            regions.append({
                "x": float(r["x"]), "y": float(r["y"]),
                "w": float(r["w"]), "h": float(r["h"]),
                "lane": int(r.get("lane", -1)),
            })
        except Exception:
            continue
    return LLMRegions(
        visualization=str(data.get("visualization", "unknown")),
        n_lanes=int(data.get("n_lanes", 0) or 0),
        regions=regions,
    )
