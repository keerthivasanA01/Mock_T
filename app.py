# app.py
"""
Mock Test Backend (Flask) — FINAL CLEAN VERSION

Features:
 - PDF Upload
 - Text Extraction
 - MCQ / 2-Mark / 13-Mark Generation using Gemini API
 - AI-Based Evaluation for 2-Mark & 13-Mark
 - Robust JSON extraction
 - Production-ready for Render
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
MAX_CONTENT_LENGTH = 20 * 1024 * 1024  # 20MB max upload

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
    """Extracts text from a PDF file."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = reader.pages[:max_pages]
    text_blocks = []

    for p in pages:
        try:
            txt = p.extract_text() or ""
        except:
            txt = ""
        if txt.strip():
            text_blocks.append(txt)

    return "\n".join(text_blocks)


def _find_first_string(obj) -> Optional[str]:
    """Recursively find first textual output from Gemini API response."""
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


def _extract_balanced_json(text: str) -> Optional[str]:
    """Extract first valid {...} block from LLM messy output."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    return None


def extract_json(text: str):
    """Parse JSON from Gemini output safely."""
    if not text or not isinstance(text, str):
        return {"raw": text}

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
# GEMINI API WRAPPER
# -------------------------
def gemini_api(prompt: str, max_tokens: int = DEFAULT_MAX_TOKENS, timeout: int = 120) -> str:
    """Calls Gemini API and returns best textual candidate."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("❌ GEMINI_API_KEY is NOT set in environment variables!")

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={api_key}"
    )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": max_tokens}
    }

    headers = {"Content-Type": "application/json"}

    last_exception = None
    for attempt in range(1, GEMINI_RETRIES + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()

            # Expected structure: candidates → content → parts → text
            try:
                cand = data.get("candidates", [])
                if cand:
                    content = cand[0].get("content", {})
                    parts = content.get("parts", [])
                    if parts:
                        part = parts[0]
                        if isinstance(part, dict) and "text" in part:
                            return part["text"]
                        text_val = _find_first_string(part)
                        if text_val:
                            return text_val

                # fallback search
                fallback = _find_first_string(data)
                return fallback or json.dumps(data)

            except Exception:
                return json.dumps(data)

        except Exception as e:
            last_exception = e
            if attempt < GEMINI_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
            else:
                break

    raise RuntimeError(f"Gemini API failed after retries: {last_exception}")


# -------------------------
# PROMPT BUILDERS
# -------------------------
def normalize_counts(user_counts: dict):
    """Normalizes counts to keys: mcq, two_mark, thirteen_mark"""
    out = {"mcq": 5, "two_mark": 5, "thirteen_mark": 2}
    if not isinstance(user_counts, dict):
        return out

    for k, v in user_counts.items():
        try:
            v = int(v)
        except:
            continue

        key = str(k).lower()

        if key.startswith("mcq"):
            out["mcq"] = v
        elif "2" in key or "two" in key:
            out["two_mark"] = v
        elif "13" in key or "thirteen" in key:
            out["thirteen_mark"] = v

    return out


def build_prompt(text, domain, subject, difficulty, language, counts):
    snippet = text[:6000]

    return f"""
You are an exam-question generator. Return STRICT JSON.

SCHEMA:
{{
  "mcq": [
    {{
      "question": "...",
      "options": ["A) ...","B) ...","C) ...","D) ..."],
      "answer": "A||explanation",
      "marks": 1
    }}
  ],
  "two_mark": [
    {{
      "question": "...",
      "answer": "...",
      "marks": 2
    }}
  ],
  "thirteen_mark": [
    {{
      "question": "...",
      "answer_outline": "...",
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

SOURCE:
{text}

Return ONLY JSON.
"""


def build_eval_prompt(q_type, question, correct, user, marks):
    return f"""
Evaluate the student's answer.

TYPE: {q_type}
MAX MARKS: {marks}

QUESTION:
{question}

CORRECT ANSWER:
{correct}

STUDENT ANSWER:
{user}

Return STRICT JSON:
{{ "score": <number>, "feedback": "<short feedback>" }}
"""


# -------------------------
# ROUTES
# -------------------------
@app.route("/", methods=["GET"])
def root():
    return {"ok": True, "service": "Mock-Test Backend"}


# ---------------------- GENERATION ----------------------
@app.route("/generate", methods=["POST"])
def generate():
    if "file" not in request.files:
        return {"error": "Upload a PDF using field 'file'"}, 400

    file = request.files["file"]
    if not allowed_file(file.filename):
        return {"error": "Only PDF allowed"}, 400

    pdf_bytes = file.read()

    try:
        text = extract_text(pdf_bytes)
    except Exception as e:
        return {"error": "Failed to extract text", "details": str(e)}, 500

    if not text.strip():
        return {"error": "PDF has no extractable text"}, 400

    domain = request.form.get("domain", "General")
    subject = request.form.get("subject", "General")
    difficulty = request.form.get("difficulty", "medium")
    language = request.form.get("language", "English")

    # Parse counts
    try:
        raw_counts = json.loads(request.form.get("counts", "{}"))
    except:
        raw_counts = {}

    counts = normalize_counts(raw_counts)

    # Build prompt & call LLM
    prompt = build_prompt(text, domain, subject, difficulty, language, counts)

    try:
        llm_out = gemini_api(prompt)
    except Exception as e:
        return {"error": "LLM call failed", "details": str(e)}, 500

    result = extract_json(llm_out)

    return {"ok": True, "result": result}


# ---------------------- EVALUATION ----------------------
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

    for i, q in enumerate(mcqs):
        correct_raw = q.get("answer", "")
        user_choice = ua_mcq[i].strip().upper() if i < len(ua_mcq) else ""

        correct = correct_raw.split("||")[0].strip().upper()

        score = 1 if user_choice == correct else 0

        total_score += score
        max_score += 1

        explanation = ""
        if "||" in correct_raw:
            explanation = correct_raw.split("||")[1]

        details.append({
            "type": "mcq",
            "question": q.get("question", ""),
            "correct": correct,
            "user": user_choice,
            "score": score,
            "explanation": explanation
        })

    # ---------- 2-Mark ----------
    two_mark = questions.get("two_mark", [])
    ua_two = user_answers.get("two_mark", [])

    for i, q in enumerate(two_mark):
        correct = q.get("answer", "")
        ans = ua_two[i] if i < len(ua_two) else ""
        marks = int(q.get("marks", 2))

        if use_ai:
            prompt = build_eval_prompt("2-mark", q["question"], correct, ans, marks)
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

    # ---------- 13-Mark ----------
    th_mark = questions.get("thirteen_mark", [])
    ua_th = user_answers.get("thirteen_mark", [])

    for i, q in enumerate(th_mark):
        correct = q.get("answer_outline", "")
        ans = ua_th[i] if i < len(ua_th) else ""
        marks = int(q.get("marks", 13))

        if use_ai:
            prompt = build_eval_prompt("13-mark", q["question"], correct, ans, marks)
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
            feedback = "Rule-based disabled for 13-mark"

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
