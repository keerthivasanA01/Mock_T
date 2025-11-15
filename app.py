# app.py
"""
Mock Test Backend (Flask) — FINAL VERSION WITH CORS + /debugkey
Fully supports frontend (React/Vite) calls.
"""

import os
import io
import json
import re
import time
from typing import Optional
from flask import Flask, request, jsonify
from flask_cors import CORS
from PyPDF2 import PdfReader
import requests

# -------------------------
# CONFIG
# -------------------------
ALLOWED_EXTENSIONS = {"pdf"}
MAX_CONTENT_LENGTH = 20 * 1024 * 1024  # 20MB

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
DEFAULT_MAX_TOKENS = 1200
GEMINI_RETRIES = 3
RETRY_DELAY = 1.5

# -------------------------
# FLASK + CORS
# -------------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})     # <---- IMPORTANT
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


# -------------------------
# HELPERS
# -------------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text(pdf_bytes: bytes, max_pages: int = 40) -> str:
    """Extract text from PDF pages."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = reader.pages[:max_pages]
    extracted = []

    for p in pages:
        try:
            txt = p.extract_text() or ""
        except:
            txt = ""
        if txt.strip():
            extracted.append(txt)

    return "\n".join(extracted)


def _find_first_string(obj) -> Optional[str]:
    """Find first string inside nested Gemini output."""
    if obj is None:
        return None
    if isinstance(obj, str):
        return obj
    if isinstance(obj, list):
        for v in obj:
            s = _find_first_string(v)
            if s:
                return s
    if isinstance(obj, dict):
        for v in obj.values():
            s = _find_first_string(v)
            if s:
                return s
    return None


def _extract_json_block(text: str) -> Optional[str]:
    """Extract balanced {...} JSON block."""
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    return None


def extract_json(text: str):
    """Parse JSON safely."""
    if not isinstance(text, str):
        return {"raw": text}

    try:
        return json.loads(text)
    except:
        block = _extract_json_block(text)
        if block:
            try:
                return json.loads(block)
            except:
                pass

    return {"raw": text}


# -------------------------
# GEMINI API WRAPPER
# -------------------------
def gemini_api(prompt: str, max_tokens: int, timeout: int = 120) -> str:
    """Call Gemini API with retry."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("❌ GEMINI_API_KEY is missing! Set it in Render Dashboard.")

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={api_key}"
    )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": max_tokens}
    }
    headers = {"Content-Type": "application/json"}

    last_error = None
    for attempt in range(1, GEMINI_RETRIES + 1):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()

            # Standard extraction
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    part0 = parts[0]
                    if isinstance(part0, dict) and "text" in part0:
                        return part0["text"]
                    text = _find_first_string(part0)
                    if text:
                        return text

            # fallback
            fallback = _find_first_string(data)
            return fallback or json.dumps(data)

        except Exception as e:
            last_error = e
            if attempt < GEMINI_RETRIES:
                time.sleep(RETRY_DELAY * attempt)

    raise RuntimeError(f"Gemini API failed after retries: {last_error}")


# -------------------------
# PROMPT BUILDERS
# -------------------------
def normalize_counts(raw):
    out = {"mcq": 5, "two_mark": 5, "thirteen_mark": 2}
    if not isinstance(raw, dict):
        return out

    for k, v in raw.items():
        try:
            v = int(v)
        except:
            continue
        key = str(k).lower()
        if "mcq" in key:
            out["mcq"] = v
        elif "2" in key or "two" in key:
            out["two_mark"] = v
        elif "13" in key or "thirteen" in key:
            out["thirteen_mark"] = v
    return out


def build_gen_prompt(text, domain, subject, difficulty, language, counts):
    return f"""
Generate exam questions STRICTLY in JSON:

SCHEMA:
{{
  "mcq": [
    {{
      "question": "",
      "options": ["A) ...","B) ...","C) ...","D) ..."],
      "answer": "A||explanation",
      "marks": 1
    }}
  ],
  "two_mark": [
    {{
      "question": "",
      "answer": "",
      "marks": 2
    }}
  ],
  "thirteen_mark": [
    {{
      "question": "",
      "answer_outline": "",
      "marks": 13
    }}
  ]
}}

Generate:
- {counts["mcq"]} MCQs
- {counts["two_mark"]} two-mark
- {counts["thirteen_mark"]} thirteen-mark

Domain: {domain}
Subject: {subject}
Difficulty: {difficulty}
Language: {language}

SOURCE CONTENT:
{text}

RETURN ONLY JSON.
"""


def build_eval_prompt(q_type, question, correct, user, marks):
    return f"""
Evaluate the answer in STRICT JSON.

QUESTION TYPE: {q_type}
MAX MARKS: {marks}

QUESTION:
{question}

CORRECT ANSWER:
{correct}

STUDENT ANSWER:
{user}

Return:
{{"score": number, "feedback": "short feedback"}}
"""


# -------------------------
# ROUTES
# -------------------------

@app.route("/")
def home():
    return {"ok": True, "service": "Mock Test Backend"}


@app.route("/debugkey")
def debugkey():
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return {"GEMINI_API_KEY": None, "message": "❌ Not Loaded"}
    return {"GEMINI_API_KEY": key[:6] + "********", "message": "✅ Key Loaded"}


# ----------------------
# GENERATE QUESTIONS
# ----------------------
@app.route("/generate", methods=["POST"])
def generate():
    if "file" not in request.files:
        return {"error": "Upload a PDF as 'file'"}, 400

    file = request.files["file"]
    if not allowed_file(file.filename):
        return {"error": "Only PDF allowed"}, 400

    pdf_bytes = file.read()

    try:
        text = extract_text(pdf_bytes)
    except Exception as e:
        return {"error": "PDF parse failed", "details": str(e)}, 500

    if not text.strip():
        return {"error": "No extractable text found in PDF"}, 400

    domain = request.form.get("domain", "General")
    subject = request.form.get("subject", "General")
    difficulty = request.form.get("difficulty", "medium")
    language = request.form.get("language", "English")

    try:
        raw_counts = json.loads(request.form.get("counts", "{}"))
    except:
        raw_counts = {}
    counts = normalize_counts(raw_counts)

    prompt = build_gen_prompt(text, domain, subject, difficulty, language, counts)

    try:
        llm_out = gemini_api(prompt, DEFAULT_MAX_TOKENS)
    except Exception as e:
        return {"error": "LLM error", "details": str(e)}, 500

    return {"ok": True, "result": extract_json(llm_out)}


# ----------------------
# EVALUATE ANSWERS
# ----------------------
@app.route("/evaluate", methods=["POST"])
def evaluate():
    data = request.get_json()
    if not data:
        return {"error": "Expected JSON body"}, 400

    questions = data.get("questions", {})
    answers = data.get("user_answers", {})
    use_ai = data.get("use_ai", True)

    total = 0
    max_total = 0
    details = []

    # ---------- MCQ ----------
    mcqs = questions.get("mcq", [])
    mcq_ans = answers.get("mcq", [])

    for i, q in enumerate(mcqs):
        correct = q.get("answer", "").split("||")[0].strip().upper()
        user = mcq_ans[i].strip().upper() if i < len(mcq_ans) else ""

        score = 1 if user == correct else 0
        total += score
        max_total += 1

        details.append({
            "type": "mcq",
            "question": q.get("question", ""),
            "correct": correct,
            "user": user,
            "score": score
        })

    # ---------- 2-MARK ----------
    twos = questions.get("two_mark", [])
    tw_ans = answers.get("two_mark", [])

    for i, q in enumerate(twos):
        correct = q.get("answer", "")
        ans = tw_ans[i] if i < len(tw_ans) else ""
        marks = int(q.get("marks", 2))

        if use_ai:
            prompt = build_eval_prompt("2-mark", q["question"], correct, ans, marks)
            try:
                parsed = extract_json(gemini_api(prompt, 512))
                score = int(parsed.get("score", 0))
                fb = parsed.get("feedback", "")
            except:
                score = 0
                fb = "AI failed"
        else:
            score = marks if ans.lower() in correct.lower() else 0
            fb = "Rule-based"

        total += score
        max_total += marks

        details.append({
            "type": "two_mark",
            "question": q.get("question", ""),
            "score": score,
            "feedback": fb
        })

    # ---------- 13-MARK ----------
    th = questions.get("thirteen_mark", [])
    th_ans = answers.get("thirteen_mark", [])

    for i, q in enumerate(th):
        correct = q.get("answer_outline", "")
        ans = th_ans[i] if i < len(th_ans) else ""
        marks = int(q.get("marks", 13))

        if use_ai:
            prompt = build_eval_prompt("13-mark", q["question"], correct, ans, marks)
            try:
                parsed = extract_json(gemini_api(prompt, 800))
                score = int(parsed.get("score", 0))
                fb = parsed.get("feedback", "")
            except:
                score = 0
                fb = "AI failed"
        else:
            score = 0
            fb = "Rule-based disabled"

        total += score
        max_total += marks

        details.append({
            "type": "13_mark",
            "question": q.get("question", ""),
            "score": score,
            "feedback": fb
        })

    return {
        "ok": True,
        "score": total,
        "max_score": max_total,
        "details": details
    }


# -------------------------
# RUN SERVER
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
