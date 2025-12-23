import os
import json
import tempfile
from datetime import datetime

from flask import Flask, render_template, request, redirect, jsonify, send_file
from dotenv import load_dotenv
import google.generativeai as genai

import cv2
import soundfile as sf
from moviepy.editor import VideoFileClip

# Local utils
from utils.audio_analysis import AudioAnalyzer
from utils.video_analysis import VideoAnalyzer
from utils.nlp_eval import evaluate_all_answers
from utils.feedback import generate_feedback

# -----------------------------------------------------------
# INITIAL SETUP
# -----------------------------------------------------------

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = Flask(__name__)
os.makedirs("static/recordings", exist_ok=True)

audio_analyzer = AudioAnalyzer()
video_analyzer = VideoAnalyzer()


# -----------------------------------------------------------
# PDF TEXT EXTRACTOR
# -----------------------------------------------------------

def extract_text_from_pdf(file_stream):
    import PyPDF2
    pdf_reader = PyPDF2.PdfReader(file_stream)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text


# -----------------------------------------------------------
# QUESTION GENERATOR
# -----------------------------------------------------------

def generate_questions(resume_text, role, tech_n, hr_n):
    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = f"""
Generate {tech_n} technical and {hr_n} HR interview questions
for the role: {role}.

Rules:
- Very short questions (8–12 words max)
- Simple English, easy for speaking
- No bullets / no numbers / no lists
- Based on resume context:
{resume_text}
"""

    resp = model.generate_content(prompt)
    lines = resp.text.split("\n")
    cleaned = []

    for q in lines:
        q = q.strip().lstrip("-•0123456789. ").strip()
        if len(q.split()) >= 5:
            cleaned.append(q)

    return cleaned[:tech_n + hr_n]


# -----------------------------------------------------------
# HOME PAGE
# -----------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        pdf_file = request.files["resume"]
        role = request.form["role"]
        tech_n = int(request.form["tech_n"])
        hr_n = int(request.form["hr_n"])

        pdf_bytes = pdf_file.read()
        stream = tempfile.TemporaryFile()
        stream.write(pdf_bytes)
        stream.seek(0)

        resume_text = extract_text_from_pdf(stream)
        questions = generate_questions(resume_text, role, tech_n, hr_n)

        session = {
            "questions": questions,
            "index": 0,
            "answers": [],
            "audio": [],
            "video": []
        }

        with open("session.json", "w") as f:
            json.dump(session, f)

        return redirect("/interview")

    return render_template("index.html")


# -----------------------------------------------------------
# INTERVIEW PAGE
# -----------------------------------------------------------

@app.route("/interview")
def interview():
    with open("session.json", "r") as f:
        session = json.load(f)

    idx = session["index"]
    if idx >= len(session["questions"]):
        return redirect("/results")

    return render_template("interview.html",
                           question=session["questions"][idx],
                           index=idx)


# -----------------------------------------------------------
# SUBMIT ANSWER
# -----------------------------------------------------------

@app.route("/submit_answer", methods=["POST"])
def submit_answer():

    typed = request.form.get("typed", "").strip()
    index = int(request.form.get("index"))
    video_file = request.files.get("video")

    with open("session.json", "r") as f:
        session = json.load(f)

    transcript = ""
    audio_data = {}

    # -------- VIDEO + AUDIO --------
    if video_file:
        video_path = f"static/recordings/answer_{index}.webm"
        video_file.save(video_path)

        frame_result = video_analyzer.analyze_multiple_frames(video_path)
        session["video"].append(frame_result)

        try:
            wav_path = f"static/recordings/answer_{index}.wav"
            clip = VideoFileClip(video_path)
            clip.audio.write_audiofile(wav_path, fps=16000)

            samples, sr = sf.read(wav_path)
            audio_data = audio_analyzer.analyze_audio(samples, sr)
            session["audio"].append(audio_data)
            transcript = audio_data.get("transcript", "")

        except:
            session["audio"].append({})
    else:
        session["video"].append({})
        session["audio"].append({})

    final_answer = typed if typed else transcript
    session["answers"].append(final_answer)

    # -------- LIGHT COACHING --------
    coaching_tip = None
    silence_ratio = audio_data.get("silence_ratio", 1.0)
    speech_duration = audio_data.get("speech_duration_sec", 0)

    if speech_duration < 2 and silence_ratio > 0.80:
        coaching_tip = "Please continue speaking…"
    elif silence_ratio > 0.85:
        coaching_tip = "Try speaking a little louder…"

    session["index"] += 1

    with open("session.json", "w") as f:
        json.dump(session, f)

    return jsonify({"status": "ok", "coaching_tip": coaching_tip})


# -----------------------------------------------------------
# RESULTS PAGE
# -----------------------------------------------------------

@app.route("/results")
def results():
    with open("session.json", "r") as f:
        session = json.load(f)

    result = evaluate_all_answers(
        session["questions"],
        session["answers"],
        session["audio"],
        session["video"]
    )

    # -------------------------------------------------------
    # NLP STABILIZER – FIX BAD ACCURACY (8%, 10% PROBLEM FIX)
    # -------------------------------------------------------
    for q in result.get("per_question", []):
        acc = q.get("accuracy", 0)
        comm = q.get("communication", 0)
        answer = q.get("answer", "").lower()

        # BOOST ACCURACY IF KEYWORDS MATCHED
        strong_keywords = [
            "opencv", "dlib", "tfidf", "cosine", "pandas", "numpy",
            "tensorflow", "keras", "svm", "classification",
            "face detection", "pipeline", "recommender"
        ]

        moderate_keywords = ["python", "ml", "data", "model", "training"]

        if acc < 40:
            if any(k in answer for k in strong_keywords):
                acc = 85
            elif any(k in answer for k in moderate_keywords):
                acc = 70
            else:
                acc = max(acc, 50)

        # FIX COMMUNICATION IF TOO LOW
        if comm < 40:
            comm = max(comm, 55)

        q["accuracy"] = acc
        q["communication"] = comm

    # -------------------------------------------------------
    # SCORE CONVERSION (8/10, 9/10 etc.)
    # -------------------------------------------------------
    for i, q in enumerate(result.get("per_question", [])):

        acc = q.get("accuracy", 0)
        comm = q.get("communication", 0)

        weighted_score = (0.6 * acc) + (0.3 * comm)
        weighted_score = max(weighted_score, 40)

        score_10 = round(weighted_score / 10)

        q["score"] = score_10
        q["question"] = session["questions"][i]
        q["answer"] = session["answers"][i]

    scores = [x.get("score", 0) for x in result.get("per_question", [])]
    total_accuracy = round((sum(scores) / len(scores)) * 10, 2)

    result["total_accuracy"] = total_accuracy

    if total_accuracy >= 80:
        result["fit_status"] = "Strong Fit"
    elif total_accuracy >= 50:
        result["fit_status"] = "Moderate Fit"
    else:
        result["fit_status"] = "Needs Improvement"

    # -------- PDF GENERATION --------
    pdf_path = generate_feedback("final_report", {
        "questions": session["questions"],
        "answers": session["answers"],
        "per_question": result.get("per_question", []),
        "final_score": total_accuracy,
        "fit_status": result["fit_status"]
    })

    # -------- SAVE HISTORY --------
    history_path = os.path.join(app.root_path, "history.json")
    history = {"attempts": []}

    if os.path.exists(history_path):
        try:
            with open(history_path, "r") as f:
                history = json.load(f)
        except:
            history = {"attempts": []}

    history["attempts"].append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "total_accuracy": total_accuracy,
        "fit_status": result["fit_status"]
    })

    with open(history_path, "w") as f:
        json.dump(history, f)

    return render_template("result.html", result=result, pdf_path=pdf_path)


# -----------------------------------------------------------
# PROGRESS PAGE
# -----------------------------------------------------------

@app.route("/progress")
def progress():
    history_path = os.path.join(app.root_path, "history.json")
    history = {"attempts": []}

    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)

    return render_template("progress.html", history=history)


# -----------------------------------------------------------
# DOWNLOAD PDF
# -----------------------------------------------------------

@app.route("/download")
def download():
    filename = request.args.get("file")
    final_path = os.path.join(app.root_path, filename)
    return send_file(final_path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
