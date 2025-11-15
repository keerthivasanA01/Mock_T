# app.py
"""
Debuggable Mock Test Backend (Flask)
- Adds better logging for Gemini failures
- Adds /debugkey to verify GEMINI_API_KEY presence
- CORS enabled
- Keeps generation and evaluation logic
"""

import os
import io
import json
import time
import traceback
from typing import Optional
from flask import Flask, request, jsonify
from flask_cors import CORS
from PyPDF2 import PdfReader
import requests

# -------------------------
# CONFIG
# -------------------------
ALLOWED_EXTENSIONS = {"pdf"}
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
MAX_TOKENS = 1200
RETRIES = 3
RETRY_DELAY = 1.3

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20MB

# -------------------------
# HELPERS
# -------------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    texts = []
    for p in reader.pages[:40]:
        try:
            t = p.extract_text() or ""
        except Exception:
            t = ""
        if t.strip():
            texts.append(t)
    return "\n".join(texts)

def find_any_text(obj) -> Optional[str]:
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

def extract_valid_json(text: str):
    if not isinstance(text, str):
        return {"raw": str(text)}
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    if start == -1:
        return {"raw": text}
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i+1])
                except Exception:
                    return {"raw": text}
    return {"raw": text}

# -------------------------
# GEMINI CALL (improved logging)
# -------------------------
def call_gemini(prompt: str, max_tokens: int = MAX_TOKENS):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in environment variables.")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": max_tokens}
    }
    headers = {"Content-Type": "application/json"}

    last_exception = None
    for attempt in range(1, RETRIES + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            # log status for debugging
            if resp.status_code != 200:
                # log response body to server logs (safe — does not print API key)
                print(f"[Gemini] NON-200 status {resp.status_code} on attempt {attempt}")
                print("[Gemini] response.text:", resp.text[:2000])
                resp.raise_for_status()
            data = resp.json()
            # try extracting common candidate text
            cands = data.get("candidates", [])
            if cands:
                content = cands[0].get("content", {})
                parts = content.get("parts", [])
                if parts:
                    part0 = parts[0]
                    if isinstance(part0, dict) and "text" in part0:
                        return part0["text"]
                    txt = find_any_text(part0)
                    if txt:
                        return txt
            # fallback to any string in response
            return find_any_text(data) or json.dumps(data)
        except Exception as e:
            last_exception = e
            # print stacktrace for Render logs
            print(f"[Gemini] attempt {attempt} failed with exception:")
            traceback.print_exc()
            if attempt < RETRIES:
                time.sleep(RETRY_DELAY)
    # after retries
    raise RuntimeError(f"Gemini API failed after {RETRIES} attempts. Last error: {last_exception}")

# -------------------------
# PROMPT BUILDING
# -------------------------
def normalize_counts(raw):
    out = {"mcq": 5, "two_mark": 5, "thirteen_mark": 2}
    if not isinstance(raw, dict):
        return out
    for k, v in raw.items():
        try:
            v = int(v)
        except Exception:
            continue
        lk = str(k).lower()
        if "mcq" in lk:
            out["mcq"] = v
        elif "2" in lk or "two" in lk:
            out["two_mark"] = v
        elif "13" in lk or "thirteen" in lk:
            out["thirteen_mark"] = v
    return out

def gen_prompt(text, domain, subject, difficulty, language, counts):
    snippet = (text or "")[:6000]
    return f"""
Generate exam questions strictly in JSON.

SCHEMA: ...
Generate:
- {counts['mcq']} MCQs
- {counts['two_mark']} two-mark
- {counts['thirteen_mark']} thirteen-mark

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

Return JSON: {{"score":0-{marks},"feedback":"short notes"}}
"""

# -------------------------
# ROUTES
# -------------------------
@app.route("/")
def root():
    return {"ok": True, "service": "mock-test-backend"}

# debug route to see if API key is loaded (only first 6 chars shown)
@app.route("/debugkey")
def debugkey():
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return {"GEMINI_API_KEY": None, "message": "NOT LOADED"}, 200
    return {"GEMINI_API_KEY": key[:6] + "********", "message": "Loaded"}, 200

@app.route("/generate", methods=["POST"])
def generate():
    try:
        if "file" not in request.files:
            return {"error": "Upload PDF as 'file'"}, 400
        f = request.files["file"]
        if not allowed_file(f.filename):
            return {"error": "Only PDF allowed"}, 400
        pdf_bytes = f.read()
        try:
            text = extract_text(pdf_bytes)
        except Exception as e:
            print("[generate] PDF extraction error:", e)
            traceback.print_exc()
            return {"error": "PDF extraction failed", "details": str(e)}, 500
        if not text.strip():
            return {"error": "No text found in PDF"}, 400

        domain = request.form.get("domain", "General")
        subject = request.form.get("subject", "General")
        difficulty = request.form.get("difficulty", "medium")
        language = request.form.get("language", "English")

        try:
            raw_counts = json.loads(request.form.get("counts", "{}"))
        except Exception:
            raw_counts = {}
        counts = normalize_counts(raw_counts)

        prompt = gen_prompt(text, domain, subject, difficulty, language, counts)

        try:
            llm_text = call_gemini(prompt)
        except Exception as e:
            print("[generate] Gemini call failed:", e)
            traceback.print_exc()
            return {"error": "LLM failed", "details": str(e)}, 500

        parsed = extract_valid_json(llm_text)
        return {"ok": True, "result": parsed}
    except Exception as e:
        print("[generate] Unexpected Error:", e)
        traceback.print_exc()
        return {"error": "Unexpected server error", "details": str(e)}, 500

@app.route("/evaluate", methods=["POST"])
def evaluate():
    try:
        data = request.get_json()
        if not data:
            return {"error": "Expected JSON body"}, 400
        questions = data.get("questions", {})
        user_answers = data.get("user_answers", {})
        use_ai = data.get("use_ai", True)

        total = 0
        max_total = 0
        details = []

        # MCQ (safe handling)
        mcqs = questions.get("mcq", [])
        ua_mcq = user_answers.get("mcq", [])
        for i, q in enumerate(mcqs):
            try:
                correct = q.get("answer", "").split("||")[0].strip().upper()
            except Exception:
                correct = ""
            user = (ua_mcq[i] if i < len(ua_mcq) else "").strip().upper()
            score = 1 if user == correct and correct else 0
            total += score
            max_total += int(q.get("marks", 1)) if isinstance(q.get("marks", int)) else 1
            details.append({"type":"mcq","question": q.get("question",""), "correct": correct, "user": user, "score": score})

        # 2-mark and 13-mark simplified (similar to earlier code)
        two_list = questions.get("two_mark", [])
        ua_two = user_answers.get("two_mark", [])
        for i, q in enumerate(two_list):
            correct = q.get("answer","")
            user = ua_two[i] if i < len(ua_two) else ""
            marks = int(q.get("marks", 2)) if isinstance(q.get("marks"), int) else 2
            if use_ai:
                try:
                    parsed = extract_valid_json(call_gemini(eval_prompt("2-mark", q.get("question",""), correct, user, marks), 600))
                    score = int(parsed.get("score", 0))
                    fb = parsed.get("feedback","")
                except Exception as e:
                    print("[evaluate] 2-mark AI error:", e)
                    traceback.print_exc()
                    score = 0
                    fb = "AI error"
            else:
                score = marks if user.lower() in correct.lower() else 0
                fb = "rule-based"
            total += score
            max_total += marks
            details.append({"type":"2_mark","question": q.get("question",""), "score": score, "feedback": fb})

        th_list = questions.get("thirteen_mark", [])
        ua_th = user_answers.get("thirteen_mark", [])
        for i, q in enumerate(th_list):
            correct = q.get("answer_outline","")
            user = ua_th[i] if i < len(ua_th) else ""
            marks = int(q.get("marks", 13)) if isinstance(q.get("marks"), int) else 13
            if use_ai:
                try:
                    parsed = extract_valid_json(call_gemini(eval_prompt("13-mark", q.get("question",""), correct, user, marks), 900))
                    score = int(parsed.get("score", 0))
                    fb = parsed.get("feedback","")
                except Exception as e:
                    print("[evaluate] 13-mark AI error:", e)
                    traceback.print_exc()
                    score = 0
                    fb = "AI error"
            else:
                score = 0
                fb = "AI disabled"
            total += score
            max_total += marks
            details.append({"type":"13_mark","question": q.get("question",""), "score": score, "feedback": fb})

        return {"ok": True, "score": total, "max_score": max_total, "details": details}
    except Exception as e:
        print("[evaluate] Unexpected error:", e)
        traceback.print_exc()
        return {"error": "Unexpected server error", "details": str(e)}, 500

# -------------------------
# RUN
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
