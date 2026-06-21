#!/usr/bin/env python3
"""本地 CLI: 对单张图跑全自动流水线, 打印 JSON, 可选导出调试图。

  纯 OpenCV:   python run.py <image> [--debug out.png]
  AI 正畸+检测: python run.py <image> --use-ai --or-model <model> [--debug out.png]
                (key 取自环境变量 OPENROUTER_API_KEY, 或 cv/.env 里的同名项)
"""
import argparse
import json
import os
import sys

import cv2

from chromalog_cv.config import Config
from chromalog_cv.pipeline import run_pipeline
from chromalog_cv import llm_detect as L
from chromalog_cv.server import _rectify


def _load_env_file():
    """简易加载脚本同目录下的 .env (KEY=VALUE), 不覆盖已存在的环境变量。"""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.isfile(env_path):
        return
    for line in open(env_path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="输入图片路径")
    ap.add_argument("--debug", help="导出调试叠加图路径")
    ap.add_argument("--use-ai", action="store_true", help="启用 OpenRouter 找板正畸 + 斑点粗框")
    ap.add_argument("--or-model", default=os.environ.get("OPENROUTER_MODEL"),
                    help="OpenRouter 视觉模型 id, 如 google/gemini-2.0-flash-001")
    args = ap.parse_args()

    _load_env_file()
    key = os.environ.get("OPENROUTER_API_KEY")

    img = cv2.imread(args.image)
    if img is None:
        print(json.dumps({"error": f"无法读取图片: {args.image}"}, ensure_ascii=False))
        sys.exit(1)

    cfg = Config()

    # ① 正畸: AI 找板(粗) + OpenCV 精修四角; 失败/未启用 -> 纯 OpenCV (与 server 同款逻辑)
    use_ai = args.use_ai
    if use_ai and not (key and args.or_model):
        print("[警告] --use-ai 已开启, 但缺少 OPENROUTER_API_KEY 或 --or-model, "
              "将回退纯 OpenCV。", file=sys.stderr)
    rec, engine, warns = _rectify(img, cfg, use_ai, args.or_model, key)

    # ② AI 斑点粗框 (同一张正畸图); 失败/未启用 -> OpenCV 兜底
    llm_regions = None
    if use_ai and key and args.or_model:
        try:
            reg = L.detect_regions(rec.image, key, args.or_model)
            llm_regions = reg.regions
            engine = "ai+opencv"
        except L.LLMError as e:
            warns.append(f"AI 斑点检测不可用, 回退 OpenCV: {e}")

    result, debug_img, rect_img = run_pipeline(
        img, cfg, debug=bool(args.debug),
        llm_regions=llm_regions, engine_used=engine, rect=rec,
    )

    payload = result.to_json()
    if isinstance(payload, dict):
        payload.setdefault("warnings", [])
        for w in warns:
            if w not in payload["warnings"]:
                payload["warnings"].append(w)
        payload["engine_used"] = engine
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"[engine_used] {engine}", file=sys.stderr)

    if args.debug and debug_img is not None:
        cv2.imwrite(args.debug, debug_img)
        rect_path = args.debug.rsplit(".", 1)[0] + "_rectified.png"
        cv2.imwrite(rect_path, rect_img)
        print(f"[调试图已保存] {args.debug}\n[正畸图已保存] {rect_path}", file=sys.stderr)


if __name__ == "__main__":
    main()