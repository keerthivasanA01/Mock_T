# app.py
"""
Mock Test Backend with OCR Support
----------------------------------
Now supports:
 - Normal PDF text extraction (PyPDF2)
 - Scanned PDFs using OCR (Tesseract + pdf2image)
 - MCQ / 2-mark / 13-mark generation
 - AI evaluation
"""

import os
import io
import json
import time
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from PyPDF2 import PdfReader
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import requests

# -----------------------------------
# CONFIG
# -----------------------------------
ALLOWED_EXTENSIONS = {"pdf"}
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
MAX_TOKENS = 1200
MAX_PAGES = 30
RETRIES = 3
RETRY_DELAY = 1.2

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# -----------------------------------
# HELPERS
# -----------------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() == "pdf"


def extract_text_pypdf2(pdf_bytes):
    """Try normal PDF text extraction."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = reader.pages[:MAX_PAGES]
        texts = []
        for p in pages:
            t = p.extract_text() or ""
            if t.strip():
                texts.append(t)
        return "\n".join(texts)
    except:
        return ""


def extract_text_ocr(pdf_bytes):
    """OCR scanned PDFs."""
    try:
        images = convert_from_bytes(pdf_bytes, dpi=200)
        text_parts = []
        for img in images[:MAX_PAGES]:
            gray = img.convert("L")  # grayscale
            t = pytesseract.image_to_string(gray)
            if t.strip():
                text_parts.append(t)
        return "\n".join(text_parts)
    except Exception as e:
        print("OCR failed:", e)
        return ""


def extract_text(pdf_bytes):
    """Use PyPDF2 → if empty → OCR."""
    text = extract_text_pypdf2(pdf_bytes)
    if text.strip():
        return text

    print(">>> Falling back to OCR (PDF appears to be scanned)")
    ocr_text = extract_text_ocr(pdf_bytes)
    return ocr_text


def find_text(obj):
    if isinstance(obj, str):
        return obj
    if isinstance(obj, list):
        for x in obj:
            r = find_text(x)
            if r:
                return r
    if isinstance(obj, dict):
        for v in obj.values():
            r = find_text(v)
            if r:
                return r
    return None


def extract_valid_json(text):
    try:
        return json.loads(text)
    except:
        pass
    if "{" not in text:
        return {"raw": text}
    start = text.find("{")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{": depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i+1])
                except:
                    return {"raw": text}
    return {"raw": text}


# -----------------------------------
# GEMINI API CALL
# -----------------------------------
def call_gemini(prompt, max_tokens=MAX_TOKENS):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY missing in environment.")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}"

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": max_tokens}
    }

    headers = {"Content-Type": "application/json"}

    last_err = None
    for attempt in range(RETRIES):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=120)
            if r.status_code != 200:
                print("Gemini non-200:", r.status_code, r.text)
                r.raise_for_status()

            data = r.json()
            txt = find_text(data)
            return txt or json.dumps(data)

        except Exception as e:
            last_err = e
            traceback.print_exc()
            time.sleep(RETRY_DELAY)

    raise RuntimeError(str(last_err))


# -----------------------------------
# PROMPTS
# -----------------------------------
def gen_prompt(text, domain, subject, difficulty, language, c):
    snippet = text[:5000]
    return f"""
Generate exam questions in strict JSON:

{{
  "mcq": [{{"question":"","options":[],"answer":"","marks":1}}],
  "two_mark": [{{"question":"","answer":"","marks":2}}],
  "thirteen_mark": [{{"question":"","answer_outline":"","marks":13}}]
}}

Generate:
- {c["mcq"]} MCQs
- {c["two_mark"]} 2-mark
- {c["thirteen_mark"]} 13-mark

Domain: {domain}
Subject: {subject}
Difficulty: {difficulty}
Language: {language}

CONTENT:
{snippet}

Return ONLY JSON.
"""


def eval_prompt(q_type, question, correct, user, marks):
    return f"""
Evaluate:

Q: {question}
Correct: {correct}
User: {user}

Return JSON:
{{"score":0-{marks},"feedback":"short notes"}}
"""


def normalize_counts(raw):
    out = {"mcq": 5, "two_mark": 5, "thirteen_mark": 2}
    if not isinstance(raw, dict):
        return out
    for k, v in raw.items():
        try:
            v = int(v)
        except:
            continue
        key = k.lower()
        if "mcq" in key:
            out["mcq"] = v
        elif "2" in key or "two" in key:
            out["two_mark"] = v
        elif "13" in key or "thirteen" in key:
            out["thirteen_mark"] = v
    return out


# -----------------------------------
# ROUTES
# -----------------------------------
@app.route("/")
def home():
    return {"ok": True, "service": "mock-test-ocr"}


@app.route("/generate", methods=["POST"])
def generate():
    if "file" not in request.files:
        return {"error": "Upload PDF as 'file'"}, 400

    f = request.files["file"]
    pdf_bytes = f.read()

    text = extract_text(pdf_bytes)

    if not text.strip():
        return {"error": "Sorry, OCR could not read your PDF."}, 400

    domain = request.form.get("domain", "General")
    subject = request.form.get("subject", "General")
    difficulty = request.form.get("difficulty", "medium")
    language = request.form.get("language", "English")

    try:
        raw = json.loads(request.form.get("counts", "{}"))
    except:
        raw = {}

    counts = normalize_counts(raw)

    prompt = gen_prompt(text, domain, subject, difficulty, language, counts)

    try:
        llm_text = call_gemini(prompt)
        result = extract_valid_json(llm_text)
    except Exception as e:
        return {"error": "LLM failed", "details": str(e)}, 500

    return {"ok": True, "result": result}


@app.route("/evaluate", methods=["POST"])
def evaluate():
    data = request.get_json()
    if not data:
        return {"error": "Expected JSON body"}, 400

    questions = data.get("questions", {})
    user_ans = data.get("user_answers", {})
    use_ai = data.get("use_ai", True)

    total = 0
    max_total = 0
    details = []

    # MCQ evaluation
    mcqs = questions.get("mcq", [])
    ua_mcq = user_ans.get("mcq", [])

    for i, q in enumerate(mcqs):
        correct = q["answer"].split("||")[0].strip().upper()
        user = ua_mcq[i].strip().upper() if i < len(ua_mcq) else ""
        score = 1 if user == correct else 0
        total += score
        max_total += 1
        details.append({"type": "mcq", "question": q["question"], "score": score})

    # Two-mark evaluation
    two = questions.get("two_mark", [])
    ua_two = user_ans.get("two_mark", [])

    for i, q in enumerate(two):
        question = q["question"]
        correct = q["answer"]
        user = ua_two[i] if i < len(ua_two) else ""
        marks = 2

        if use_ai:
            parsed = extract_valid_json(
                call_gemini(eval_prompt("2-mark", question, correct, user, marks), 500)
            )
            score = int(parsed.get("score", 0))
        else:
            score = marks if user.lower() in correct.lower() else 0

        total += score
        max_total += marks
        details.append({"type": "two_mark", "question": question, "score": score})

    # Thirteen-mark evaluation
    th = questions.get("thirteen_mark", [])
    ua_th = user_ans.get("thirteen_mark", [])

    for i, q in enumerate(th):
        question = q["question"]
        correct = q["answer_outline"]
        user = ua_th[i] if i < len(ua_th) else ""
        marks = 13

        parsed = extract_valid_json(
            call_gemini(eval_prompt("13-mark", question, correct, user, marks), 900)
        )
        score = int(parsed.get("score", 0))

        total += score
        max_total += marks
        details.append({"type": "13_mark", "question": question, "score": score})

    return {"ok": True, "score": total, "max_score": max_total, "details": details}


# -----------------------------------
# RUN SERVER
# -----------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
