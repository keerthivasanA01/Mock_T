# app.py
"""
Mock Test Backend (Flask) — FINAL VERSION WITH /debugkey
 - Uses GEMINI_API_KEY from Render env (NO HARDCODED KEY)
 - Generates MCQ / 2-Mark / 13-Mark
 - Evaluates answers using Gemini AI
 - Includes /debugkey route to verify API key is loaded
"""

import os
import io
import json
import re
import time
from typing import Optional
from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
import requests

# -------------------------
# CONFIG
# -------------------------
ALLOWED_EXTENSIONS = {"pdf"}
MAX_CONTENT_LENGTH = 20 * 1024 * 1024  # 20MB max upload size

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
DEFAULT_MAX_TOKENS = 1200
GEMINI_RETRIES = 3
RETRY_DELAY = 1.5

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


# -------------------------
# HELPERS
# -------------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text(pdf_bytes: bytes, max_pages: int = 40) -> str:
    """Extracts text from PDF."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    texts = []

    for page in reader.pages[:max_pages]:
        try:
            txt = page.extract_text() or ""
        except:
            txt = ""
        if txt.strip():
            texts.append(txt)

    return "\n".join(texts)


def _find_first_string(obj) -> Optional[str]:
    """Search through Gemini JSON to find first string text."""
    if obj is None:
        return None
    if isinstance(obj, str):
        return obj
    if isinstance(obj, list):
        for x in obj:
            s = _find_first_string(x)
            if s:
                return s
    if isinstance(obj, dict):
        for v in obj.values():
            s = _find_first_string(v)
            if s:
                return s
    return None


def _extract_balanced_json(text: str) -> Optional[str]:
    """Extracts first {...} valid block."""
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
    """Parse LLM output safely."""
    if not isinstance(text, str):
        return {"raw": text}

    # direct attempt
    try:
        return json.loads(text)
    except:
        block = _extract_balanced_json(text)
        if block:
            try:
                return json.loads(block)
            except:
                pass

    return {"raw": text}


# -------------------------
# GEMINI API CALL
# -------------------------
def gemini_api(prompt: str, max_tokens: int = DEFAULT_MAX_TOKENS, timeout: int = 120) -> str:
    """Calls Gemini using environment API key."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("❌ GEMINI_API_KEY is NOT SET in environment variables.")

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
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()

            data = resp.json()

            # candidate structure
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    part0 = parts[0]
                    if isinstance(part0, dict) and "text" in part0:
                        return part0["text"]
                    txt = _find_first_string(part0)
                    if txt:
                        return txt

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
        k = str(k).lower()
        if "mcq" in k:
            out["mcq"] = v
        elif "2" in k or "two" in k:
            out["two_mark"] = v
        elif "13" in k or "thirteen" in k:
            out["thirteen_mark"] = v
    return out


def build_prompt(text, domain, subject, difficulty, language, counts):
    snippet = text[:6000]

    return f"""
Generate exam questions STRICTLY in JSON:

SCHEMA:
{{
  "mcq": [
    {{
      "question": "",
      "options": ["A) ...","B) ...","C) ...","D) ..."],
      "answer": "A||short explanation",
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

Return ONLY JSON.
"""


def build_eval_prompt(q_type, question, correct, user, marks):
    return f"""
Evaluate the answer.

TYPE: {q_type}
MAX MARKS: {marks}

QUESTION:
{question}

CORRECT ANSWER:
{correct}

STUDENT ANSWER:
{user}

Return STRICT JSON:
{{ "score": <0-{marks}>, "feedback": "..." }}
"""


# -------------------------
# ROUTES
# -------------------------
@app.route("/")
def home():
    return {"ok": True, "service": "Mock Test Backend"}


# TEMP DEBUG ROUTE (remove after testing)
@app.route("/debugkey")
def debugkey():
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return {"GEMINI_API_KEY": None, "message": "❌ NOT LOADED"}, 200
    return {"GEMINI_API_KEY": key[:6] + "********", "message": "✅ Loaded"}, 200


# ---------------------- GENERATE ------------------------
@app.route("/generate", methods=["POST"])
def generate():
    if "file" not in request.files:
        return {"error": "Upload PDF as 'file'"}, 400

    f = request.files["file"]
    if not allowed_file(f.filename):
        return {"error": "Only PDF allowed"}, 400

    pdf_bytes = f.read()

    try:
        text = extract_text(pdf_bytes)
    except Exception as e:
        return {"error": "PDF extraction failed", "details": str(e)}, 500

    if not text.strip():
        return {"error": "No text in PDF"}, 400

    # preferences
    domain = request.form.get("domain", "General")
    subject = request.form.get("subject", "General")
    difficulty = request.form.get("difficulty", "medium")
    language = request.form.get("language", "English")

    # counts
    try:
        raw_counts = json.loads(request.form.get("counts", "{}"))
    except:
        raw_counts = {}

    counts = normalize_counts(raw_counts)

    # Prompt
    prompt = build_prompt(text, domain, subject, difficulty, language, counts)

    try:
        llm_out = gemini_api(prompt)
    except Exception as e:
        return {"error": "LLM call failed", "details": str(e)}, 500

    parsed = extract_json(llm_out)

    return {"ok": True, "result": parsed}


# ---------------------- EVALUATE ------------------------
@app.route("/evaluate", methods=["POST"])
def evaluate():
    data = request.get_json()
    if not data:
        return {"error": "Expected JSON"}, 400

    questions = data.get("questions", {})
    user_answers = data.get("user_answers", {})
    use_ai = data.get("use_ai", True)

    total_score = 0
    max_score = 0
    details = []

    # ---------- MCQ ----------
    mcqs = questions.get("mcq", [])
    ua_mcq = user_answers.get("mcq", [])

    for idx, q in enumerate(mcqs):
        correct_raw = q.get("answer", "")
        correct_letter = correct_raw.split("||")[0].strip().upper()
        user_letter = ua_mcq[idx].strip().upper() if idx < len(ua_mcq) else ""

        score = 1 if user_letter == correct_letter else 0

        total_score += score
        max_score += 1

        details.append({
            "type": "mcq",
            "question": q.get("question", ""),
            "correct": correct_letter,
            "user": user_letter,
            "score": score
        })

    # ---------- 2-MARK ----------
    two_mark = questions.get("two_mark", [])
    ua_two = user_answers.get("two_mark", [])

    for idx, q in enumerate(two_mark):
        correct = q.get("answer", "")
        ans = ua_two[idx] if idx < len(ua_two) else ""

        marks = int(q.get("marks", 2))

        if use_ai:
            prompt = build_eval_prompt("2-mark", q.get("question", ""), correct, ans, marks)
            try:
                llm_out = gemini_api(prompt)
                parsed = extract_json(llm_out)
                score = int(parsed.get("score", 0))
                feedback = parsed.get("feedback", "")
            except:
                score = 0
                feedback = "AI scoring failed"
        else:
            score = marks if ans.lower() in correct.lower() else 0
            feedback = "Rule-based scoring"

        total_score += score
        max_score += marks

        details.append({
            "type": "2_mark",
            "question": q.get("question", ""),
            "score": score,
            "feedback": feedback
        })

    # ---------- 13-MARK ----------
    thirteen_mark = questions.get("thirteen_mark", [])
    ua_thirteen = user_answers.get("thirteen_mark", [])

    for idx, q in enumerate(thirteen_mark):
        correct = q.get("answer_outline", "")
        ans = ua_thirteen[idx] if idx < len(ua_thirteen) else ""
        marks = int(q.get("marks", 13))

        if use_ai:
            prompt = build_eval_prompt("13-mark", q.get("question", ""), correct, ans, marks)
            try:
                llm_out = gemini_api(prompt, max_tokens=800)
                parsed = extract_json(llm_out)
                score = int(parsed.get("score", 0))
                feedback = parsed.get("feedback", "")
            except:
                score = 0
                feedback = "AI scoring failed"
        else:
            score = 0
            feedback = "13-mark requires AI scoring"

        total_score += score
        max_score += marks

        details.append({
            "type": "13_mark",
            "question": q.get("question", ""),
            "score": score,
            "feedback": feedback
        })

    return {
        "ok": True,
        "score": total_score,
        "max_score": max_score,
        "details": details
    }


# -------------------------
# RUN SERVER
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
