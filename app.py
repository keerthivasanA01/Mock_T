# app.py
"""
FULL BACKEND WITH:
 - PDF Upload
 - Text Extraction
 - MCQ / 2-Mark / 13-Mark Generation
 - AI-Based Evaluation of User Answers (Gemini)
"""

import os
import io
import re
import json
import time
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
import requests

# -------------------------
# CONFIG
# -------------------------
ALLOWED_EXTENSIONS = {"pdf"}
MAX_CONTENT_LENGTH = 20 * 1024 * 1024  # 20MB PDF limit
GEMINI_MODEL = "gemini-1.5-flash"
DEFAULT_MAX_TOKENS = 1200

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


# -------------------------
# UTILITIES
# -------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text(pdf_bytes, max_pages=40):
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = reader.pages[:max_pages]
    text = []
    for p in pages:
        try:
            txt = p.extract_text() or ""
        except:
            txt = ""
        if txt:
            text.append(txt)
    return "\n".join(text)


def gemini_api(prompt, max_tokens=1200):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in environment variables")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": max_tokens}
    }

    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()

    data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]
    except:
        return str(data)


def extract_json(text):
    try:
        return json.loads(text)
    except:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except:
                pass
    return {"raw": text}


# -------------------------
# BUILD PROMPT FOR QUESTION GENERATION
# -------------------------
def build_generation_prompt(text, domain, subject, difficulty, language, counts):
    snippet = text[:6000]

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


# -------------------------
# BUILD PROMPT FOR AI EVALUATION
# -------------------------
def build_eval_prompt(q_type, question, correct_answer, user_answer, marks):
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
{{
  "score": <number>,
  "feedback": "<short feedback>"
}}
"""


# -------------------------
# ROUTES
# -------------------------

@app.route("/", methods=["GET"])
def health():
    return jsonify({"ok": True, "service": "mock-test-backend"})


# -------------------------
# 1️⃣ GENERATION ENDPOINT
# -------------------------
@app.route("/generate", methods=["POST"])
def generate():
    if "file" not in request.files:
        return jsonify({"error": "No PDF uploaded"}), 400

    file = request.files["file"]
    if not allowed_file(file.filename):
        return jsonify({"error": "Upload a PDF only"}), 400

    pdf_bytes = file.read()
    text = extract_text(pdf_bytes)

    domain = request.form.get("domain", "General")
    subject = request.form.get("subject", "General")
    difficulty = request.form.get("difficulty", "medium")
    language = request.form.get("language", "English")

    counts_raw = request.form.get("counts", "")
    try:
        counts = json.loads(counts_raw)
    except:
        counts = {"mcq": 5, "two_mark": 5, "thirteen_mark": 2}

    counts.setdefault("mcq", 5)
    counts.setdefault("two_mark", 5)
    counts.setdefault("thirteen_mark", 2)

    prompt = build_generation_prompt(text, domain, subject, difficulty, language, counts)

    llm_out = gemini_api(prompt)
    parsed = extract_json(llm_out)

    return jsonify({"ok": True, "result": parsed})


# -------------------------
# 2️⃣ EVALUATION ENDPOINT
# -------------------------
@app.route("/evaluate", methods=["POST"])
def evaluate():
    data = request.get_json()

    questions = data.get("questions", {})
    user_answers = data.get("user_answers", {})
    use_ai = data.get("use_ai", True)

    total_score = 0
    max_score = 0
    details = []

    # ----------- MCQ -----------
    if "mcq" in questions:
        q_list = questions["mcq"]
        user_list = user_answers.get("mcq", [])
        for i, q in enumerate(q_list):
            correct = q["answer"].split("||")[0].strip()
            user = user_list[i].strip().upper() if i < len(user_list) else ""

            is_correct = (user == correct)
            score = 1 if is_correct else 0
            explanation = q["answer"].split("||")[1] if "||" in q["answer"] else ""

            total_score += score
            max_score += 1

            details.append({
                "type": "mcq",
                "question": q["question"],
                "correct_option": correct,
                "user_option": user,
                "score": score,
                "explanation": explanation
            })

    # ----------- 2-MARK (AI evaluation) -----------
    if "two_mark" in questions:
        q_list = questions["two_mark"]
        user_list = user_answers.get("two_mark", [])

        for i, q in enumerate(q_list):
            correct_ans = q["answer"]
            user_ans = user_list[i] if i < len(user_list) else ""
            marks = 2

            if use_ai:
                prompt = build_eval_prompt("2-mark", q["question"], correct_ans, user_ans, marks)
                llm = gemini_api(prompt)
                res = extract_json(llm)
                score = res.get("score", 0)
                feedback = res.get("feedback", "")
            else:
                # Simple matching
                score = 2 if user_ans.lower() in correct_ans.lower() else 0
                feedback = "Keyword-based evaluation (AI disabled)"

            total_score += score
            max_score += marks

            details.append({
                "type": "2_mark",
                "question": q["question"],
                "user_answer": user_ans,
                "score": score,
                "feedback": feedback
            })

    # ----------- 13-MARK (AI evaluation) -----------
    if "thirteen_mark" in questions:
        q_list = questions["thirteen_mark"]
        user_list = user_answers.get("thirteen_mark", [])

        for i, q in enumerate(q_list):
            correct_outline = q["answer_outline"]
            user_ans = user_list[i] if i < len(user_list) else ""
            marks = 13

            if use_ai:
                prompt = build_eval_prompt("13-mark", q["question"], correct_outline, user_ans, marks)
                llm = gemini_api(prompt, max_tokens=800)
                res = extract_json(llm)
                score = res.get("score", 0)
                feedback = res.get("feedback", "")
            else:
                score = 0
                feedback = "AI evaluation disabled"

            total_score += score
            max_score += marks

            details.append({
                "type": "13_mark",
                "question": q["question"],
                "user_answer": user_ans,
                "score": score,
                "feedback": feedback
            })

    return jsonify({
        "ok": True,
        "score": total_score,
        "max_score": max_score,
        "details": details
    })


# -------------------------
# RUN
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
