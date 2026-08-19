// ================================
// SecureCode Analyzer - Frontend
// ================================

const SAMPLES = {
  sql: `def get_user(username):
    query = "SELECT * FROM users WHERE name = '" + username + "'"
    cursor.execute(query)
    return cursor.fetchone()`,

  xss: `function showName(name) {
  document.getElementById("greeting").innerHTML = "Hello " + name;
}`,

  safe: `import bcrypt

def hash_password(password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))`
};

// Sample buttons binding
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".sample-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const key = btn.dataset.sample;
      if (SAMPLES[key]) document.getElementById("code").value = SAMPLES[key];
    });
  });
});

// Score color helper
function getScoreColor(score) {
  if (score >= 80) return "#10b981"; // green
  if (score >= 50) return "#f59e0b"; // amber
  return "#ef4444";                  // red
}

// Animate progress ring
function updateScoreCircle(score) {
  const ring = document.getElementById("progressRing");
  if (!ring) return;
  const radius = 68;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (score / 100) * circumference;
  ring.style.strokeDasharray = circumference;
  ring.style.strokeDashoffset = offset;
  ring.style.stroke = getScoreColor(score);
}

// Reset all UI fields
function resetUI() {
  document.getElementById("issues").innerHTML = "";
  document.getElementById("rawjson").textContent = "{}";
  document.getElementById("lang").textContent = "-";
  document.getElementById("score").textContent = "-";
  document.getElementById("score").style.color = "";
  document.getElementById("prediction").textContent = "-";
  document.getElementById("confidence").textContent = "-";
  document.getElementById("engineUsed").textContent = "-";
  document.getElementById("engineBadge").textContent = "-";

  const ring = document.getElementById("progressRing");
  if (ring) {
    ring.style.strokeDashoffset = 427;
    ring.style.stroke = "#6366f1";
  }

  const chip = document.getElementById("predictionChip");
  chip.classList.remove("prediction-secure", "prediction-vulnerable");
}

// Main analyze function
async function analyze() {
  const code = document.getElementById("code").value.trim();

  // Get engine from radio
  const engineRadio = document.querySelector('input[name="engine"]:checked');
  const engine = engineRadio ? engineRadio.value : "hybrid";

  const btn = document.getElementById("btn");
  const btnText = document.getElementById("btnText");
  const spinner = document.getElementById("spinner");
  const status = document.getElementById("status");

  if (!code) {
    status.textContent = "Please enter or paste code first.";
    status.className = "status-text error";
    return;
  }

  resetUI();
  btn.disabled = true;
  spinner.classList.remove("hidden");
  btnText.textContent = "Analyzing...";
  status.textContent = "";
  status.className = "status-text";

  try {
    const res = await fetch("/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code, engine })
    });

    const data = await res.json();

    if (!res.ok) throw new Error(data.error || "Request failed");

    // Score
    const score = Number(data.security_score ?? 0);
    document.getElementById("score").textContent = score;
    document.getElementById("score").style.color = getScoreColor(score);
    updateScoreCircle(score);

    // Language
    document.getElementById("lang").textContent = data.detected_language || "-";

    // Engine
    const usedEngine = (data.engine || "-").toUpperCase();
    document.getElementById("engineUsed").textContent = usedEngine;
    document.getElementById("engineBadge").textContent = `Engine: ${usedEngine}`;

    // ML Prediction
    const predictionEl = document.getElementById("prediction");
    const chip = document.getElementById("predictionChip");

    if (data.ml_prediction !== undefined && data.ml_prediction !== null && data.ml_prediction !== "N/A") {
      const pred = String(data.ml_prediction).toLowerCase();
      const isVuln = ["1", "vulner", "unsafe", "weak"].some(w => pred.includes(w));
      predictionEl.textContent = isVuln ? "Vulnerable" : "Secure";
      chip.classList.add(isVuln ? "prediction-vulnerable" : "prediction-secure");
    } else {
      predictionEl.textContent = "—";
    }

    // Confidence
    if (data.ml_confidence !== null && data.ml_confidence !== undefined) {
      document.getElementById("confidence").textContent =
        (data.ml_confidence * 100).toFixed(2) + "%";
    } else {
      document.getElementById("confidence").textContent = "—";
    }

    // Issues
    const issuesEl = document.getElementById("issues");
    issuesEl.innerHTML = "";

    if (Array.isArray(data.issues) && data.issues.length > 0) {
      data.issues.forEach(item => {
        const li = document.createElement("li");
        li.textContent = item;
        const lower = String(item).toLowerCase();
        if (lower.includes("no strong") || lower.includes("no issue")) {
          li.classList.add("no-issues");
        }
        issuesEl.appendChild(li);
      });
    } else {
      const li = document.createElement("li");
      li.classList.add("no-issues");
      li.textContent = "No security issues detected.";
      issuesEl.appendChild(li);
    }

    document.getElementById("rawjson").textContent = JSON.stringify(data, null, 2);

    status.textContent = "Analysis completed successfully.";
    status.className = "status-text success";

  } catch (err) {
    status.textContent = "Analysis failed: " + err.message;
    status.className = "status-text error";
  } finally {
    btn.disabled = false;
    spinner.classList.add("hidden");
    btnText.textContent = "Analyze";
  }
}
