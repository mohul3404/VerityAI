let selectedDataset = null;

document.addEventListener("DOMContentLoaded", () => {
  const input = document.querySelector("#datasetFile");
  const zone = document.querySelector("#dropZone");
  input.addEventListener("change", () => selectFile(input.files[0]));
  ["dragenter", "dragover"].forEach((name) => zone.addEventListener(name, (event) => {
    event.preventDefault(); zone.classList.add("dragging");
  }));
  ["dragleave", "drop"].forEach((name) => zone.addEventListener(name, (event) => {
    event.preventDefault(); zone.classList.remove("dragging");
  }));
  zone.addEventListener("drop", (event) => selectFile(event.dataTransfer.files[0]));
  document.querySelector("#removeFile").addEventListener("click", (event) => {
    event.preventDefault();
    selectedDataset = null;
    input.value = "";
    document.querySelector("#selectedFile").classList.add("d-none");
    document.querySelector("#uploadTrainButton").disabled = true;
  });
  document.querySelector("#uploadTrainButton").addEventListener("click", uploadAndTrain);
});

function selectFile(file) {
  if (!file) return;
  if (!file.name.toLowerCase().endsWith(".csv")) {
    App.showToast("Please choose a CSV file.", "error");
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    App.showToast("CSV exceeds the 10 MB limit.", "error");
    return;
  }
  selectedDataset = file;
  document.querySelector("#fileName").textContent = file.name;
  document.querySelector("#fileSize").textContent = formatBytes(file.size);
  document.querySelector("#selectedFile").classList.remove("d-none");
  document.querySelector("#uploadTrainButton").disabled = false;
  App.renderIcons();
}

async function uploadAndTrain() {
  if (!selectedDataset) return;
  const form = new FormData();
  form.append("file", selectedDataset);
  document.querySelector("#trainingProgress").classList.remove("d-none");
  App.loading(true, "Validating dataset and training models...");
  try {
    const payload = await App.request("/api/upload", { method: "POST", body: form });
    document.querySelectorAll(".progress-step").forEach((step) => step.classList.add("active"));
    App.showToast(`${payload.message} ${payload.metrics.selected_model} selected.`);
  } catch (error) {
    App.showToast(error.message, "error");
  } finally {
    App.loading(false);
  }
}

function formatBytes(bytes) {
  if (!bytes) return "0 bytes";
  const units = ["bytes", "KB", "MB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), 2);
  return `${(bytes / (1024 ** index)).toFixed(index ? 1 : 0)} ${units[index]}`;
}
