import os
import json
import re
import pickle
from pathlib import Path

from flask import Flask, render_template, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv

# =========================
# Setup
# =========================
load_dotenv()
app = Flask(__name__)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

BASE_DIR = Path(__file__).resolve().parent



MODEL_PATH = r"C:\stack\results_logistic_regression\logistic_regression_model.pkl"
VECTORIZER_PATH = r"C:\stack\results_logistic_regression\tfidf_vectorizer.pkl"

# Load local ML artifacts
with open(MODEL_PATH, "rb") as f:
    ML_MODEL = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    TFIDF_VECTORIZER = pickle.load(f)

INSTRUCTIONS = """
You are a senior AppSec code reviewer.
Return STRICT JSON only (no extra text).
Be concise: short issues only, no long descriptions.
Detect the programming language.
Compute security_score 0-100 (higher is more secure).
List key issues found (max 8) as short phrases.
"""

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "detected_language": {"type": "string"},
        "security_score": {"type": "integer", "minimum": 0, "maximum": 100},
        "issues": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 8
        }
    },
    "required": ["detected_language", "security_score", "issues"]
}


# =========================
# Helpers
# =========================
def clean_code_text(code: str) -> str:
    if not code:
        return ""
    code = str(code)
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    code = code.replace("\t", "    ")
    code = re.sub(r"[ \f\v]+", " ", code)
    code = re.sub(r"\n{3,}", "\n\n", code)
    code = code.strip()
    if len(code) > 20000:
        code = code[:20000]
    return code


def detect_language_simple(code: str) -> str:
    c = (code or "").lower()

    if "<?php" in c:
        return "PHP"
    if "#include" in c:
        return "C/C++"
    if "using namespace std" in c:
        return "C++"
    if "public class" in c or "system.out.println" in c:
        return "Java"
    if "def " in c and "import " in c:
        return "Python"
    if "function " in c or "console.log" in c or "=>" in c:
        return "JavaScript"
    if "select " in c and " from " in c:
        return "SQL"

    return "Unknown"


def normalize_label(label) -> str:
    return str(label).strip().lower()


def is_vulnerable_label(label) -> bool:
    label = normalize_label(label)

    safe_words = {"0", "safe", "benign", "clean", "non-vulnerable", "not vulnerable"}
    vuln_words = {"1", "vulnerable", "unsafe", "buggy", "weak", "flaw"}

    if label in safe_words:
        return False
    if label in vuln_words:
        return True

    return any(word in label for word in ["vulner", "unsafe", "weak", "flaw", "security"])


# =========================
# OpenAI Analysis
# =========================
def analyze_code_openai(code: str) -> dict:
    code = clean_code_text(code)

    if not code:
        return {
            "engine": "openai",
            "detected_language": "Unknown",
            "security_score": 0,
            "issues": ["Empty code input"]
        }

    if client is None:
        raise RuntimeError("Missing OPENAI_API_KEY")

    resp = client.responses.create(
        model="gpt-4o-mini",
        instructions=INSTRUCTIONS,
        input=f"Analyze the following code for security issues (keep it short):\n\n{code}",
        text={
            "format": {
                "type": "json_schema",
                "name": "security_report",
                "schema": SCHEMA,
                "strict": True
            }
        },
        store=False
    )

    result = json.loads(resp.output_text)
    result["engine"] = "openai"
    return result


# =========================
# Local ML Analysis
# =========================
def analyze_code_ml(code: str) -> dict:
    code = clean_code_text(code)

    if not code:
        return {
            "engine": "ml",
            "detected_language": "Unknown",
            "security_score": 0,
            "issues": ["Empty code input"],
            "ml_prediction": "N/A",
            "ml_confidence": None
        }

    X = TFIDF_VECTORIZER.transform([code])
    pred = ML_MODEL.predict(X)[0]
    pred_str = str(pred)

    confidence = None
    security_score = 50

    # If calibrated model supports probabilities
    if hasattr(ML_MODEL, "predict_proba"):
        probs = ML_MODEL.predict_proba(X)[0]
        classes = [str(c) for c in ML_MODEL.classes_]
        class_probs = dict(zip(classes, probs))

        confidence = float(max(probs))

        vuln_probs = [
            float(prob) for cls, prob in class_probs.items()
            if is_vulnerable_label(cls)
        ]

        if vuln_probs:
            vuln_prob = max(vuln_probs)
            security_score = int(round((1 - vuln_prob) * 100))
        else:
            # fallback if no explicit vulnerable label found
            if is_vulnerable_label(pred_str):
                security_score = int(round((1 - confidence) * 100))
            else:
                security_score = int(round(confidence * 100))
    else:
        # fallback if no probability support
        if is_vulnerable_label(pred_str):
            security_score = 20
        else:
            security_score = 80

    issues = [f"ML prediction: {pred_str}"]
    if is_vulnerable_label(pred_str):
        issues.append("Potential vulnerable pattern detected")
    else:
        issues.append("No strong vulnerable pattern detected")

    return {
        "engine": "ml",
        "detected_language": detect_language_simple(code),
        "security_score": max(0, min(100, security_score)),
        "issues": issues[:8],
        "ml_prediction": pred_str,
        "ml_confidence": confidence
    }


# =========================
# Hybrid Analysis
# =========================
def analyze_code_hybrid(code: str) -> dict:
    ml_result = analyze_code_ml(code)
    ai_result = analyze_code_openai(code)

    merged_issues = []
    for item in ml_result.get("issues", []) + ai_result.get("issues", []):
        if item not in merged_issues:
            merged_issues.append(item)

    final_score = round((0.6 * ml_result["security_score"]) + (0.4 * ai_result["security_score"]))

    return {
        "engine": "hybrid",
        "detected_language": (
            ai_result.get("detected_language")
            if ai_result.get("detected_language") != "Unknown"
            else ml_result.get("detected_language", "Unknown")
        ),
        "security_score": final_score,
        "issues": merged_issues[:8],
        "ml_prediction": ml_result.get("ml_prediction"),
        "ml_confidence": ml_result.get("ml_confidence"),
        "ml_score": ml_result.get("security_score"),
        "ai_score": ai_result.get("security_score")
    }


# =========================
# Routes
# =========================
@app.get("/")
def home():
    return render_template("home.html")


@app.get("/analyzer")
def analyzer_page():
    return render_template("analyzer.html")


@app.post("/analyze")
def analyze_api():
    data = request.get_json(silent=True) or {}
    code = (data.get("code") or "").strip()
    engine = (data.get("engine") or "hybrid").strip().lower()

    if not code:
        return jsonify({"error": "Missing code"}), 400

    if len(code) > 20000:
        return jsonify({"error": "Code too large (max 20,000 chars)"}), 413

    try:
        if engine == "ml":
            result = analyze_code_ml(code)
        elif engine == "openai":
            result = analyze_code_openai(code)
        elif engine == "hybrid":
            result = analyze_code_hybrid(code)
        else:
            return jsonify({"error": "Invalid engine"}), 400

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": "Analysis failed", "details": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)