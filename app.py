import os
from flask import Flask, render_template, request
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from google import genai

app = Flask(__name__)

# Initialize Gemini Flash client
GENAI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCccFzVDJHDP07AvJLSqwyrXOP7XwGzGA0")
client = genai.Client(api_key=GENAI_API_KEY)

# Download required NLTK data quietly
for pkg in ("punkt", "stopwords", "punkt_tab", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"):
    nltk.download(pkg, quiet=True)

def generate_nltk_summary(text: str) -> str:
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    filtered = [w for w in words if w.isalnum() and w not in stop_words]

    freq = FreqDist(filtered)
    scores = {s: sum(freq[word] for word in word_tokenize(s.lower())) for s in sentences}
    summary = " ".join(sorted(scores, key=scores.get, reverse=True)[:2])

    pos = nltk.pos_tag(word_tokenize(text))
    verbs = {w for w, t in pos if t.startswith("VB")}
    nouns = {w for w, t in pos if t.startswith("NN")}

    return (
        f"Sentences: {len(sentences)}\n"
        f"NLTK Summary: {summary}\n"
        f"Verbs: {', '.join(sorted(verbs))}\n"
        f"Nouns: {', '.join(sorted(nouns))}"
    )

def refine_with_gen_ai(nltk_summary: str, original_text: str) -> str:
    prompt = (
        f"Problem Statement: {original_text}\n"
        f"Initial Analysis: {nltk_summary}\n\n"
        "Produce a detailed, professional report with the following sections:\n"
        "Impact:\n"
        "- Quantify SLA breach (e.g., current vs target latency)\n"
        "Root Cause:\n"
        "- Explain inefficient SQL patterns and missing indexes\n"
        "Recommendations:\n"
        "- Specify indexes to create, query rewrites, caching strategies, partitioning\n"
        "Expected Results:\n"
        "- Predicted performance improvements and SLA compliance\n"
        "Format each section with a clear heading and concise bullet points."
    )
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt]
    )
    return response.text.strip()

@app.route("/", methods=["GET", "POST"])
def index():
    refined_output = None
    statement = ""
    if request.method == "POST":
        statement = request.form.get("problem_statement", "").strip()
        if statement:
            nltk_summary = generate_nltk_summary(statement)
            refined_output = refine_with_gen_ai(nltk_summary, statement)
    return render_template("index.html", refined_output=refined_output, problem_statement=statement)

if __name__ == "__main__":
    app.run(debug=True)
