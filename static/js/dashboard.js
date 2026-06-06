let leaderboard = [];
let comparisonChart;
let datasetPage = 1;
let datasetPages = 1;
let searchTimer;

document.addEventListener("modelStatus", (event) => renderDashboard(event.detail.metrics));
document.addEventListener("DOMContentLoaded", () => {
  document.querySelector("#metricSelector").addEventListener("change", renderComparisonChart);
  document.querySelector("#retrainButton").addEventListener("click", retrain);
  document.querySelector("#datasetSearch").addEventListener("input", () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => { datasetPage = 1; loadDataset(); }, 280);
  });
  document.querySelector("#datasetSort").addEventListener("change", () => { datasetPage = 1; loadDataset(); });
  document.querySelector("#prevPage").addEventListener("click", () => { if (datasetPage > 1) { datasetPage -= 1; loadDataset(); } });
  document.querySelector("#nextPage").addEventListener("click", () => { if (datasetPage < datasetPages) { datasetPage += 1; loadDataset(); } });
  loadDataset();
});

function renderDashboard(metrics) {
  leaderboard = metrics.leaderboard || [];
  document.querySelector("#selectedModel").textContent = metrics.selected_model;
  document.querySelector("#precisionMetric").textContent = App.percent(metrics.precision);
  document.querySelector("#recallMetric").textContent = App.percent(metrics.recall);
  document.querySelector("#datasetSize").textContent = Number(metrics.dataset_size).toLocaleString();
  const matrix = metrics.confusion_matrix.flat();
  ["#trueReal", "#falseFake", "#falseReal", "#trueFake"].forEach((selector, index) => {
    document.querySelector(selector).textContent = matrix[index] ?? "--";
  });
  renderComparisonChart();
}

function renderComparisonChart() {
  if (!leaderboard.length) return;
  const metric = document.querySelector("#metricSelector").value;
  const context = document.querySelector("#comparisonChart");
  comparisonChart?.destroy();
  comparisonChart = new Chart(context, {
    type: "bar",
    data: {
      labels: leaderboard.map((item) => item.model),
      datasets: [{
        data: leaderboard.map((item) => Math.round(item[metric] * 1000) / 10),
        backgroundColor: leaderboard.map((_, index) => index === 0 ? "#635bff" : "rgba(99,91,255,.28)"),
        borderRadius: 5,
        maxBarThickness: 42,
      }],
    },
    options: {
      maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: (ctx) => `${ctx.raw}%` } } },
      scales: {
        y: { beginAtZero: true, suggestedMax: 100, grid: { color: "rgba(148,163,184,.15)" }, ticks: { callback: (value) => `${value}%` } },
        x: { grid: { display: false } },
      },
    },
  });
}

async function retrain() {
  App.loading(true, "Benchmarking all classifiers...");
  try {
    const payload = await App.request("/api/train", { method: "POST" });
    renderDashboard(payload.metrics);
    App.showToast(`Training complete. ${payload.metrics.selected_model} selected.`);
  } catch (error) {
    App.showToast(error.message, "error");
  } finally {
    App.loading(false);
  }
}

async function loadDataset() {
  const search = encodeURIComponent(document.querySelector("#datasetSearch")?.value || "");
  const sort = document.querySelector("#datasetSort")?.value || "title";
  try {
    const payload = await App.request(`/api/dataset?page=${datasetPage}&page_size=8&search=${search}&sort=${sort}`);
    datasetPages = payload.pages;
    const body = document.querySelector("#datasetBody");
    body.innerHTML = "";
    payload.items.forEach((item) => {
      const row = document.createElement("tr");
      row.innerHTML = `<td>${escapeHtml(item.title || "Untitled article")}</td><td>${escapeHtml(item.text)}</td><td><span class="label-chip ${item.label}">${item.label}</span></td>`;
      body.appendChild(row);
    });
    if (!payload.items.length) body.innerHTML = `<tr><td colspan="3" class="text-center py-5">No articles match this search.</td></tr>`;
    document.querySelector("#datasetSummary").textContent = `${payload.total.toLocaleString()} articles`;
    document.querySelector("#pageIndicator").textContent = `${payload.page} / ${payload.pages}`;
  } catch (error) {
    App.showToast(error.message, "error");
  }
}

function escapeHtml(value) {
  const div = document.createElement("div");
  div.textContent = value ?? "";
  return div.innerHTML;
}
