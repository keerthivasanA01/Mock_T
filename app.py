# app.py
"""
Corrected Mock Test Backend (Flask)

Improvements:
 - Normalizes counts keys (accepts several common formats)
 - Robust Gemini API call with retries and safer candidate extraction
 - Balanced-brace JSON extraction from LLM output (less fragile than regex)
 - Defensive checks for missing fields in questions/evaluation
 - Keeps single-file layout for Render deployment

ENV:
 - GEMINI_API_KEY (required)
 - GEMINI_MODEL (optional, defaults to gemini-1.5-flash)
 - PORT (optional)
"""

import os
import io
import json
import re
import time
from typing import Tuple, Optional
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
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
GEMINI_RETRY_DELAY = 1.5

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


# -------------------------
# UTILITIES
# -------------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text(pdf_bytes: bytes, max_pages: int = 40) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = reader.pages[:max_pages]
    texts = []
    for p in pages:
        try:
            t = p.extract_text() or ""
        except Exception:
            t = ""
        if t:
            texts.append(t)
    return "\n".join(texts)


def _find_first_string(obj) -> Optional[str]:
    """Recursively find the first string in nested data structure."""
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


def gemini_api(prompt: str, max_tokens: int = DEFAULT_MAX_TOKENS, timeout: int = 120) -> str:
    """
    Call Gemini REST API (v1beta generateContent).
    Returns the textual candidate output (string).
    Raises RuntimeError on irrecoverable failures.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

    model = GEMINI_MODEL
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key=AIzaSyBb2thQ3RIayUIxn5fE7XgB_TrFjPz8Qdc"

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": max_tokens}
    }
    headers = {"Content-Type": "application/json"}

    last_exc = None
    for attempt in range(1, GEMINI_RETRIES + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()

            # try standard shape: candidates -> content -> parts -> text
            try:
                cand = data.get("candidates", [])
                if cand and isinstance(cand, list):
                    c0 = cand[0]
                    # candidate.content.parts could be list of strings or list of dicts with text
                    content = c0.get("content")
                    if content:
                        parts = content.get("parts")
                        if parts and isinstance(parts, list):
                            part0 = parts[0]
                            if isinstance(part0, str):
                                return part0
                            if isinstance(part0, dict):
                                # try known text key
                                txt = part0.get("text") or _find_first_string(part0)
                                if txt:
                                    return txt
                # fallback: find any string in response
                s = _find_first_string(data)
                if s:
                    return s
                # as last resort, return stringified json
                return json.dumps(data)
            except Exception:
                return json.dumps(data)
        except Exception as e:
            last_exc = e
            if attempt < GEMINI_RETRIES:
                time.sleep(GEMINI_RETRY_DELAY * attempt)
            else:
                break
    raise RuntimeError(f"Gemini API call failed after {GEMINI_RETRIES} attempts. Last error: {last_exc}")


def _extract_balanced_json(text: str) -> Optional[str]:
    """
    Find first balanced { ... } substring in text.
    Works by scanning for the first '{' and matching braces.
    Returns substring or None.
    """
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
                return text[start: i + 1]
    return None


def extract_json(text: str):
    """Try to parse JSON from text robustly."""
    if not text or not isinstance(text, str):
        return {"raw": text}
    # try direct load
    try:
        return json.loads(text)
    except Exception:
        # try to extract balanced JSON substring
        js = _extract_balanced_json(text)
        if js:
            try:
                return json.loads(js)
            except Exception:
                pass
    # fallback
    return {"raw": text}


# -------------------------
# PROMPT BUILDERS
# -------------------------
def _normalize_counts(raw_counts: dict) -> dict:
    """
    Normalize user-provided counts to keys:
    'mcq', 'two_mark', 'thirteen_mark'
    Accepts alternate forms: '2mark', 'two-mark', '13mark', '13-mark', 'thirteen_mark', etc.
    """
    result = {"mcq": 5, "two_mark": 5, "thirteen_mark": 2}
    if not isinstance(raw_counts, dict):
        return result
    for k, v in raw_counts.items():
        if v is None:
            continue
        key = str(k).lower().strip()
        try:
            val = int(v)
        except Exception:
            # ignore non-int values
            continue
        if key in ("mcq", "mchoice", "multiple_choice"):
            result["mcq"] = val
        elif key in ("2mark", "two_mark", "two-mark", "two mark"):
            result["two_mark"] = val
        elif key in ("13mark", "thirteen_mark", "13-mark", "13 mark", "thirteen-mark"):
            result["thirteen_mark"] = val
        else:
            # if key contains digits try heuristics
            if key.startswith("2") or "two" in key:
                result["two_mark"] = val
            if key.startswith("13") or "thirteen" in key:
                result["thirteen_mark"] = val
            if key == "mcq":
                result["mcq"] = val
    return result


def build_generation_prompt(text: str, domain: str, subject: str, difficulty: str, language: str, counts: dict) -> str:
    snippet = (text or "")[:6000]
    prompt = f"""
You are an exam-question generator. Produce STRICT JSON only.

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
- {counts['mcq']} MCQs
- {counts['two_mark']} two-mark
- {counts['thirteen_mark']} thirteen-mark

Domain: {domain}
Subject: {subject}
Difficulty: {difficulty}
Language: {language}

SOURCE CONTENT:
----------------
{snippet}
----------------

Return JSON ONLY.
"""
    return prompt


def build_eval_prompt(q_type: str, question: str, correct_answer: str, user_answer: str, marks: int) -> str:
    return f"""
Evaluate the following student's answer.

QUESTION TYPE: {q_type}
MAX MARKS: {marks}

QUESTION:
{question}

CORRECT ANSWER / OUTLINE:
{correct_answer}

STUDENT ANSWER:
{user_answer}

TASK:
1. Give a score from 0 to {marks}.
2. Provide short feedback on correctness and missing points.
3. Respond in STRICT JSON FORMAT:
{{ "score": <number>, "feedback": "<short feedback>" }}
"""


# -------------------------
# ROUTES
# -------------------------
@app.route("/", methods=["GET"])
def health():
    return jsonify({"ok": True, "service": "mock-test-backend"})


@app.route("/generate", methods=["POST"])
def generate():
    # file
    if "file" not in request.files:
        return jsonify({"error": "No file part (upload a PDF as 'file')"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed. Upload a PDF."}), 400

    # read file bytes
    try:
        pdf_bytes = file.read()
    except Exception as e:
        return jsonify({"error": "Failed to read uploaded file", "details": str(e)}), 400

    # extract text
    try:
        text = extract_text(pdf_bytes)
        if not text.strip():
            return jsonify({"error": "No extractable text found in PDF"}), 400
    except Exception as e:
        return jsonify({"error": "PDF parsing failed", "details": str(e)}), 500

    # preferences
    domain = request.form.get("domain", "General")
    subject = request.form.get("subject", "General")
    difficulty = request.form.get("difficulty", "medium")
    language = request.form.get("language", "English")

    counts_raw = request.form.get("counts", "")
    # counts may be JSON or string; try parse
    try:
        parsed_counts = json.loads(counts_raw) if counts_raw else {}
    except Exception:
        # try to accept simple csv like "mcq:5,2mark:3"
        parsed_counts = {}
        try:
            for token in (counts_raw or "").split(","):
                if ":" in token:
                    k, v = token.split(":", 1)
                    parsed_counts[k.strip()] = int(v.strip())
        except Exception:
            parsed_counts = {}

    counts = _normalize_counts(parsed_counts)

    # build prompt & call LLM
    prompt = build_generation_prompt(text=text, domain=domain, subject=subject,
                                     difficulty=difficulty, language=language, counts=counts)
    try:
        llm_out = gemini_api(prompt, max_tokens=DEFAULT_MAX_TOKENS)
    except Exception as e:
        return jsonify({"error": "LLM call failed", "details": str(e)}), 500

    parsed = extract_json(llm_out)
    return jsonify({"ok": True, "result": parsed})


@app.route("/evaluate", methods=["POST"])
def evaluate():
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "Expected JSON body"}), 400

    questions = payload.get("questions", {})
    user_answers = payload.get("user_answers", {})
    use_ai = bool(payload.get("use_ai", True))

    total_score = 0
    max_score = 0
    details = []

    # MCQ evaluation
    mcq_list = questions.get("mcq", [])
    user_mcq = user_answers.get("mcq", [])
    for idx, q in enumerate(mcq_list):
        correct_field = q.get("answer", "")
        # answer may be "A||explanation" or "A" or "A) text"
        correct_letter = ""
        if "||" in correct_field:
            correct_letter = correct_field.split("||", 1)[0].strip()
        else:
            # try to extract leading letter A/B/C/D
            m = re.match(r"^\s*([A-D])\b", correct_field, re.I)
            if m:
                correct_letter = m.group(1).upper()
            else:
                # try options to find which option is marked or fallback to empty
                correct_letter = correct_field.strip().upper()

        user_choice = ""
        if idx < len(user_mcq):
            user_choice = (user_mcq[idx] or "").strip().upper()

        score = 0
        try:
            if user_choice and correct_letter and user_choice[0] == correct_letter[0]:
                score = int(q.get("marks", 1)) if isinstance(q.get("marks"), int) else 1
        except Exception:
            score = 0

        total_score += score
        max_score += int(q.get("marks", 1)) if isinstance(q.get("marks"), int) else 1

        explanation = ""
        if "||" in correct_field:
            explanation = correct_field.split("||", 1)[1].strip()

        details.append({
            "type": "mcq",
            "question": q.get("question", ""),
            "correct_option": correct_letter,
            "user_option": user_choice,
            "score": score,
            "explanation": explanation
        })

    # 2-mark evaluation
    two_list = questions.get("two_mark", [])
    user_two = user_answers.get("two_mark", [])
    for idx, q in enumerate(two_list):
        correct_ans = q.get("answer", "")
        user_ans = user_two[idx] if idx < len(user_two) else ""
        marks = int(q.get("marks", 2)) if isinstance(q.get("marks"), int) else 2

        score = 0
        feedback = ""
        if use_ai:
            try:
                prompt = build_eval_prompt("2-mark", q.get("question", ""), correct_ans, user_ans, marks)
                llm = gemini_api(prompt, max_tokens=512)
                res = extract_json(llm)
                score = int(res.get("score", 0)) if isinstance(res.get("score", 0), (int, float)) else 0
                feedback = res.get("feedback", "") if isinstance(res.get("feedback", ""), str) else str(res.get("feedback", ""))
            except Exception as e:
                score = 0
                feedback = f"AI evaluation failed: {e}"
        else:
            # simple keyword matching
            try:
                score = marks if (user_ans and user_ans.lower() in correct_ans.lower()) else 0
                feedback = "Keyword-based evaluation"
            except Exception:
                score = 0
                feedback = "Keyword eval error"

        total_score += score
        max_score += marks
        details.append({
            "type": "2_mark",
            "question": q.get("question", ""),
            "user_answer": user_ans,
            "score": score,
            "feedback": feedback
        })

    # 13-mark evaluation
    th_list = questions.get("thirteen_mark", [])
    user_th = user_answers.get("thirteen_mark", [])
    for idx, q in enumerate(th_list):
        correct_outline = q.get("answer_outline", "")
        user_ans = user_th[idx] if idx < len(user_th) else ""
        marks = int(q.get("marks", 13)) if isinstance(q.get("marks"), int) else 13

        score = 0
        feedback = ""
        if use_ai:
            try:
                prompt = build_eval_prompt("13-mark", q.get("question", ""), correct_outline, user_ans, marks)
                llm = gemini_api(prompt, max_tokens=800)
                res = extract_json(llm)
                score = int(res.get("score", 0)) if isinstance(res.get("score", 0), (int, float)) else 0
                feedback = res.get("feedback", "") if isinstance(res.get("feedback", ""), str) else str(res.get("feedback", ""))
            except Exception as e:
                score = 0
                feedback = f"AI evaluation failed: {e}"
        else:
            score = 0
            feedback = "AI evaluation disabled"

        total_score += score
        max_score += marks
        details.append({
            "type": "13_mark",
            "question": q.get("question", ""),
            "user_answer": user_ans,
            "score": score,
            "feedback": feedback
        })

    return jsonify({"ok": True, "score": total_score, "max_score": max_score, "details": details})


# -------------------------
# RUN
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # For production, use gunicorn. For quick local test:
    app.run(host="0.0.0.0", port=port, debug=False)
