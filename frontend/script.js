const form = document.getElementById("heartForm");
const resultCard = document.getElementById("resultCard");
const resultText = document.getElementById("resultText");
const riskBar = document.getElementById("riskBar");
const clearBtn = document.getElementById("clearBtn");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  resultCard.classList.add("hidden");
  resultText.textContent = "Predicting…";

  const payload = {
    age: Number(document.getElementById("age").value),
    sex: Number(document.getElementById("sex").value),
    cp: Number(document.getElementById("cp").value),
    trestbps: Number(document.getElementById("trestbps").value),
    chol: Number(document.getElementById("chol").value),
    fbs: Number(document.getElementById("fbs").value),
    restecg: Number(document.getElementById("restecg").value),
    thalach: Number(document.getElementById("thalach").value),
    exang: Number(document.getElementById("exang").value),
    oldpeak: Number(document.getElementById("oldpeak").value),
    slope: Number(document.getElementById("slope").value),
    ca: Number(document.getElementById("ca").value),
    thal: Number(document.getElementById("thal").value)
  };

  try {
    // Call the FastAPI backend (default port 8000 from the backend instructions)
    const resp = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload)
    });

    if (!resp.ok) throw new Error("Server error");

    const data = await resp.json();
    // backend returns `risk_score` (0..1) and `risk_level`
    const prob = Number(data.risk_score);
    const percent = (prob * 100).toFixed(1);

    // Update UI
    resultText.innerHTML = `<strong>Estimated Heart Disease Probability:</strong> ${percent}%`;
    riskBar.style.width = `${Math.min(100, prob * 100)}%`;
    // color change
    if (prob < 0.33) riskBar.style.background = "linear-gradient(90deg,#4caf50,#2e7d32)";
    else if (prob < 0.66) riskBar.style.background = "linear-gradient(90deg,#ffb300,#ff9800)";
    else riskBar.style.background = "linear-gradient(90deg,#ff7043,#c62828)";

    resultCard.classList.remove("hidden");
    window.scrollTo({top: resultCard.offsetTop - 20, behavior: 'smooth'});
  } catch (err) {
    resultText.textContent = "Prediction failed. Is the backend running?";
    resultCard.classList.remove("hidden");
  }
});

clearBtn.addEventListener("click", () => {
  form.reset();
  riskBar.style.width = "0%";
  resultCard.classList.add("hidden");
});