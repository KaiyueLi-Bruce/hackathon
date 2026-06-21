"""AI 实验报告生成 (spec §10)。

两种模式:
  - generate_questions: 无实验笔记时, 让模型基于检测到的 Rf 数据提出若干澄清问题,
    交用户回答后再出报告。
  - generate_report: 据 (Rf 数据 + 条件 + 时程 + 可选实验笔记/用户答案) 生成
    固定章节的 markdown 报告。

报告模型可与图像识别模型不同 (由调用方传入 model)。纯文本 chat, 不发图。
任何失败抛 LLMError, 由端点转成 ok:false。
"""
from __future__ import annotations

import json
import urllib.request
import urllib.error

from .llm_detect import OPENROUTER_URL, LLMError, _extract_json

REPORT_GUIDE = (
    "You are a synthetic/organic chemist writing a TLC experiment report. "
    "TLC is qualitative / semi-quantitative — never invent numbers; use ONLY the data given. "
    "State confidence and the reasoning behind any judgment (e.g. 'product co-spots with the "
    "standard at the same Rf → likely the same compound'). Output GitHub-flavored Markdown with "
    "exactly these sections:\n"
    "## Purpose\n## Conditions\n## Observations\n(include a Markdown Rf table: Lane | Label | Rf)\n"
    "## Interpretation\n## Next steps\n## Notes\n"
    "Be concise and practical."
)


def _chat_text(prompt: str, api_key: str, model: str, timeout: float = 60.0) -> str:
    if not api_key:
        raise LLMError("缺少 OpenRouter API key")
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
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
    return content


def generate_questions(data: dict, api_key: str, model: str) -> list:
    """无笔记时: 基于 Rf 数据/条件提出 3-6 个简短澄清问题。返回字符串列表。"""
    prompt = (
        "A chemist ran this TLC but provided no lab notebook. Based on the detected data below, "
        "ask 3-6 SHORT clarifying questions whose answers are needed to interpret the result and "
        "write a useful report (e.g. reaction type, expected product, which lane is which, "
        "time points, visualization). Return ONLY JSON: {\"questions\": [\"...\"]}.\n\n"
        "DATA (JSON):\n" + json.dumps(data, ensure_ascii=False)
    )
    content = _chat_text(prompt, api_key, model)
    out = _extract_json(content)
    qs = out.get("questions", [])
    return [str(q) for q in qs][:8]


def generate_report(data: dict, notebook_text: str, answers: str,
                    api_key: str, model: str) -> str:
    """生成 markdown 报告。notebook_text / answers 任一可为空字符串。"""
    parts = [REPORT_GUIDE, "\n\n=== Detected TLC data (JSON) ===\n",
             json.dumps(data, ensure_ascii=False)]
    if notebook_text.strip():
        parts.append("\n\n=== Lab notebook (user-provided) ===\n" + notebook_text.strip())
    if answers.strip():
        parts.append("\n\n=== User answers to clarifying questions ===\n" + answers.strip())
    parts.append("\n\nWrite the report now.")
    return _chat_text("".join(parts), api_key, model)
