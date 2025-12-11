# AI-Powered Multimodal Interview Assistant

A virtual interview platform that evaluates candidates using text, audio, and video analysis to measure communication quality, confidence, and technical accuracy—offering a realistic AI-driven interview preparation experience.

## About
The **AI-Powered Multimodal Interview Assistant** overcomes the limitations of traditional mock interview tools that rely only on text-based responses.  
This system analyzes **spoken language, facial expressions, and textual content** to provide deeper, more accurate evaluation.

It automatically generates personalized technical and HR questions from the candidate’s resume and job role. Users respond via video/audio, and the system delivers multimodal scoring along with detailed feedback and improvement suggestions.

Designed for students, job seekers, and placement cells, this tool strengthens interview performance through continuous practice and data-driven evaluation.

## Features
- AI-generated technical & HR interview questions based on resume analysis.
- Real-time multimodal evaluation using:
  - **Text (NLP scoring with Gemini)**
  - **Audio (Whisper transcription & clarity metrics)**
  - **Video (OpenCV-based facial detection & analysis)**
- Question-wise scoring: accuracy, communication, voice tone, expression.
- Downloadable PDF performance report.
- History tracking for progress improvement.
- User-friendly web interface built with Flask.

## Requirements
- **OS:** Windows 10/11 or Ubuntu (64-bit recommended)
- **Language:** Python 3.12
- **Core AI Tools & Frameworks:**
  - Google Gemini API (NLP + question generation)
  - Faster-Whisper (speech-to-text)
  - OpenCV (video/frame processing)
- **Python Libraries:**
  - Flask(python)
  - NumPy
  - SoundFile
  - MoviePy
  - FPDF
  - Requests
  - dotenv
- **Frontend Technologies:**
  - HTML5, CSS3, JavaScript
  - MediaRecorder API for camera & mic capture
  - SpeechSynthesis API

- **Storage:**
  - Local filesystem (static/recordings, PDF, session.json)


## System Architecture

<img width="1184" height="864" alt="image" src="https://github.com/user-attachments/assets/a00e2a96-804f-421b-a82c-8f26341fc1fa" />




## Output:

### **Output 1 — Resume Upload**
Extracts text and generates personalized questions.

<img width="1915" height="836" alt="image" src="https://github.com/user-attachments/assets/dc92008c-0165-44f2-8293-e22f10d80b37" />


### **Output 2 — Interview Interface**
Records audio/video responses and displays each question.

<img width="1532" height="770" alt="image" src="https://github.com/user-attachments/assets/b1447b03-8c2e-4802-a73f-b698300d8557" />


### **Output 3 — Results Dashboard**
Shows NLP scores, multimodal accuracy, detailed feedback per question, and downloadable report.

<img width="1348" height="866" alt="image" src="https://github.com/user-attachments/assets/611e512a-594d-4823-b817-d844dd7943ba" />


### **Output 4 — Progress Tracking**
Displays historical performance improvement across multiple attempts.

<img width="1536" height="686" alt="image" src="https://github.com/user-attachments/assets/461822dc-188a-4785-8188-abf1b5b47279" />



## Results and Impact
This AI-driven system enhances interview readiness by providing **real-time, multimodal feedback** on communication, confidence, and technical accuracy.  
It offers:
sample ouput(Based on user's performance):
NLP Scores
 - Confidence: 65%
 - Communication: 70%
 - Answer Accuracy: 80%

- Deep behavioral analysis missing in text-only tools  
- Automated and unbiased scoring  
- Practical insights for improving interview performance  
- A scalable solution for students, job seekers, and training centers  

It effectively serves as a **virtual AI interview coach**, supporting long-term skill development.



## Articles published / References
1. Y.-C. Chou et al., “An AI Mock-interview Platform for Interview Performance Analysis,” ICIET 2022.  
2. P. K. Mishra et al., “AI-Driven Virtual Mock Interview Development,” SCIS&ISIS 2024.  
3. J. V. Barpute et al., “A Survey of AI-Driven Mock Interviews Using GenAI and Machine Learning (InterviewX),” ICUIS 2024.  

---
