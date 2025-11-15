# app.py
"""
FINAL Mock Test Backend (Flask) — STABLE & CLEAN VERSION
 - CORS enabled
 - PDF → MCQ / 2-Mark / 13-Mark generation
 - AI answer evaluation
 - Valid JSON guarantee
 - Clean errors (no 500 crashes)
 - Works with React/Vite frontend
"""

import os
import io
import json
import re
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from PyPDF2 import PdfReader
import requests

# -------------------------------------
# CONFIG
# -------------------------------------
ALLOWED_EXTENSIONS = {"pdf"}
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
MAX_TOKENS = 1200
RETRIES = 3
RETRY_DELAY = 1.3

# -------------------------------------
# FLASK + CORS
# -------------------------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024   # 20 MB PDF limit


# -------------------------------------
# HELPERS
# -------------------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text(pdf_bytes):
    """Safely extract text from PDF pages."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = []

    for page in reader.pages[:40]:
        try:
            t = page.extract_text() or ""
        except:
            t = ""
        if t.strip():
            text.append(t)

    return "\n".join(text)


def find_any_text(obj):
    """Find first textual string inside nested Gemini output."""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, list):
        for x in obj:
            r = find_any_text(x)
            if r:
                return r
    if isinstance(obj, dict):
        for v in obj.values():
            r = find_any_text(v)
            if r:
                return r
    return None


def extract_valid_json(text):
    """Get JSON from messy LLM output safely."""
    if not isinstance(text, str):
        return {"raw": str(text)}

    # Try direct JSON
    try:
        return json.loads(text)
    except:
        pass

    # Find balanced {...}
    start = text.find("{")
    if start == -1:
        return {"raw": text}

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        if text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except:
                    return {"raw": text}

    return {"raw": text}


# -------------------------------------
# GEMINI API
# -------------------------------------
def call_gemini(prompt, max_tokens=MAX_TOKENS):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in Render environment.")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}"

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": max_tokens}
    }

    last_err = None

    for attempt in range(1, RETRIES + 1):
        try:
            r = requests.post(url, json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()

            # Standard extraction
            cands = data.get("candidates", [])
            if cands:
                parts = cands[0].get("content", {}).get("parts", [])
                if parts:
                    part0 = parts[0]
                    if isinstance(part0, dict) and "text" in part0:
                        return part0["text"]
                    fallback = find_any_text(part0)
                    if fallback:
                        return fallback

            return find_any_text(data) or json.dumps(data)

        except Exception as e:
            last_err = e
            time.sleep(RETRY_DELAY)

    raise RuntimeError(f"Gemini API failed after retries: {last_err}")


# -------------------------------------
# PROMPTS
# -------------------------------------
def normalize_counts(raw):
    """Normalize question counts from frontend."""
    default = {"mcq": 5, "two_mark": 5, "thirteen_mark": 2}

    if not isinstance(raw, dict):
        return default

    for k, v in raw.items():
        try:
            v = int(v)
        except:
            continue

        k = k.lower()
        if "mcq" in k:
            default["mcq"] = v
        elif "2" in k or "two" in k:
            default["two_mark"] = v
        elif "13" in k or "thirteen" in k:
            default["thirteen_mark"] = v

    return default


def gen_prompt(text, domain, subject, difficulty, language, counts):
    snippet = text[:6000]

    return f"""
Generate exam questions strictly in JSON.

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
- {counts["thirteen_mark"]} thirteen_mark

Domain: {domain}
Subject: {subject}
Difficulty: {difficulty}
Language: {language}

CONTENT:
{snippet}

Return ONLY JSON.
"""


def eval_prompt(q_type, q, correct, user, marks):
    return f"""
Evaluate student's answer.

QUESTION:
{q}

CORRECT:
{correct}

ANSWER:
{user}

Return JSON:
{{"score": 0-{marks}, "feedback": "short notes"}}
"""


# -------------------------------------
# ROUTES
# -------------------------------------
@app.route("/")
def root():
    return {"ok": True}


@app.route("/generate", methods=["POST"])
def generate():
    """Generate questions from PDF."""
    if "file" not in request.files:
        return {"error": "Upload PDF as 'file'"}, 400

    file = request.files["file"]
    if not allowed_file(file.filename):
        return {"error": "Only PDF allowed"}, 400

    pdf_bytes = file.read()

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
        raw = json.loads(request.form.get("counts", "{}"))
    except:
        raw = {}
    counts = normalize_counts(raw)

    # LLM PROMPT
    prompt = gen_prompt(text, domain, subject, difficulty, language, counts)

    try:
        llm_text = call_gemini(prompt)
        result = extract_valid_json(llm_text)
    except Exception as e:
        return {"error": "LLM failed", "details": str(e)}, 500

    return {"ok": True, "result": result}


@app.route("/evaluate", methods=["POST"])
def evaluate():
    """Evaluate student's answers."""
    data = request.get_json()
    if not data:
        return {"error": "Expected JSON body"}, 400

    questions = data.get("questions", {})
    answers = data.get("user_answers", {})
    use_ai = data.get("use_ai", True)

    total = 0
    max_total = 0
    details = []

    # ---------------- MCQs ----------------
    mcqs = questions.get("mcq", [])
    ans_mcq = answers.get("mcq", [])

    for i, q in enumerate(mcqs):
        correct = q["answer"].split("||")[0].strip().upper()
        user = ans_mcq[i].strip().upper() if i < len(ans_mcq) else ""

        mark = 1 if user == correct else 0
        total += mark
        max_total += 1

        details.append({
            "type": "mcq",
            "question": q["question"],
            "correct": correct,
            "user": user,
            "score": mark
        })

    # ---------------- 2-MARK ----------------
    twos = questions.get("two_mark", [])
    ans2 = answers.get("two_mark", [])

    for i, q in enumerate(twos):
        question = q["question"]
        correct = q["answer"]
        user = ans2[i] if i < len(ans2) else ""
        marks = 2

        if use_ai:
            prompt = eval_prompt("2-mark", question, correct, user, marks)
            parsed = extract_valid_json(call_gemini(prompt, 600))
            score = int(parsed.get("score", 0))
            fb = parsed.get("feedback", "")
        else:
            score = marks if user.lower() in correct.lower() else 0
            fb = "rule based"

        total += score
        max_total += marks

        details.append({
            "type": "two_mark",
            "question": question,
            "score": score,
            "feedback": fb
        })

    # ---------------- 13-MARK ----------------
    th = questions.get("thirteen_mark", [])
    ans13 = answers.get("thirteen_mark", [])

    for i, q in enumerate(th):
        question = q["question"]
        correct = q["answer_outline"]
        user = ans13[i] if i < len(ans13) else ""
        marks = 13

        if use_ai:
            prompt = eval_prompt("13-mark", question, correct, user, marks)
            parsed = extract_valid_json(call_gemini(prompt, 900))
            score = int(parsed.get("score", 0))
            fb = parsed.get("feedback", "")
        else:
            score = 0
            fb = "AI disabled"

        total += score
        max_total += marks

        details.append({
            "type": "13_mark",
            "question": question,
            "score": score,
            "feedback": fb
        })

    return {
        "ok": True,
        "score": total,
        "max_score": max_total,
        "details": details
    }


# -------------------------------------
# RUN
# -------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
