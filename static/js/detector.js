const detectorExamples = [
  {
    headline: "Health department expands seasonal vaccination clinics",
    content: "The city health department confirmed that six additional vaccination clinics will open next week. The schedule was published on the department website and officials said appointments can be booked online."
  },
  {
    headline: "Secret miracle fruit cures every disease overnight",
    content: "Doctors are stunned and hospitals are hiding this unbelievable discovery from everyone! A viral post claims one secret fruit guarantees an instant cure, but provides no named researchers, clinical trial, or official evidence."
  },
];
let exampleIndex = 0;

document.addEventListener("modelStatus", (event) => {
  const { metrics, usage } = event.detail;
  document.querySelector("#statModel").textContent = metrics.selected_model;
  document.querySelector("#statAccuracy").textContent = App.percent(metrics.accuracy);
  document.querySelector("#statF1").textContent = App.percent(metrics.f1);
  document.querySelector("#statPredictions").textContent = Number(usage.total_predictions || 0).toLocaleString();
});

document.addEventListener("DOMContentLoaded", () => {
  const form = document.querySelector("#predictionForm");
  const content = document.querySelector("#content");
  content.addEventListener("input", () => {
    document.querySelector("#contentCounter").textContent = `${content.value.length.toLocaleString()} / 50,000`;
  });
  document.querySelector("#exampleButton").addEventListener("click", () => {
    const example = detectorExamples[exampleIndex % detectorExamples.length];
    exampleIndex += 1;
    document.querySelector("#headline").value = example.headline;
    content.value = example.content;
    content.dispatchEvent(new Event("input"));
  });
  form.addEventListener("submit", analyzeArticle);
});

async function analyzeArticle(event) {
  event.preventDefault();
  const headline = document.querySelector("#headline").value.trim();
  const content = document.querySelector("#content").value.trim();
  App.loading(true, "Analyzing language signals...");
  try {
    const result = await App.request("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ headline, content }),
    });
    renderPrediction(result);
    document.querySelector("#statPredictions").textContent = Number(result.usage.total_predictions).toLocaleString();
    App.showToast("Article analysis completed.");
  } catch (error) {
    App.showToast(error.message, "error");
  } finally {
    App.loading(false);
  }
}

function renderPrediction(result) {
  const panel = document.querySelector("#predictionResult");
  panel.classList.remove("empty", "fake-result", "real-result");
  panel.classList.add(`${result.prediction}-result`);
  document.querySelector("#resultEmpty").classList.add("d-none");
  document.querySelector("#resultContent").classList.remove("d-none");
  document.querySelector("#verdictLabel").textContent = result.label;
  document.querySelector("#resultModel").textContent = result.model;
  document.querySelector("#riskScore").textContent = `${result.risk_score.toFixed(1)}%`;
  document.querySelector("#gaugeFill").style.width = `${result.risk_score}%`;
  document.querySelector("#confidenceValue").textContent = App.percent(result.confidence);
  document.querySelector("#confidenceCircle").style.setProperty("--progress", Math.round(result.confidence * 100));
  document.querySelector("#resultSummary").textContent = result.prediction === "fake"
    ? "The model found language patterns associated with fabricated or sensational reporting. Verify the claim with primary sources."
    : "The language patterns are more consistent with reliable reporting. This is a model assessment, not independent fact verification.";

  const icon = document.querySelector("#verdictIcon");
  icon.innerHTML = `<i data-lucide="${result.prediction === "fake" ? "triangle-alert" : "shield-check"}"></i>`;
  document.querySelector("#analysisPanel").classList.remove("d-none");
  document.querySelector("#explanationMethod").textContent = result.explanation.method;
  renderIndicators("#positiveIndicators", result.explanation.positive_indicators);
  renderIndicators("#negativeIndicators", result.explanation.negative_indicators);
  document.querySelector("#wordCount").textContent = result.insights.word_count.toLocaleString();
  document.querySelector("#sentimentScore").textContent = Number(result.insights.polarity).toFixed(2);
  document.querySelector("#subjectivityScore").textContent = App.percent(result.insights.subjectivity);
  document.querySelector("#readabilityScore").textContent = result.insights.readability_score;
  App.renderIcons();
  document.querySelector("#analysisPanel").scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderIndicators(selector, items) {
  const container = document.querySelector(selector);
  container.innerHTML = "";
  if (!items.length) {
    container.innerHTML = "<span>No strong signal detected</span>";
    return;
  }
  items.forEach((item) => {
    const chip = document.createElement("span");
    chip.textContent = `${item.term} · ${item.influence.toFixed(3)}`;
    container.appendChild(chip);
  });
}
