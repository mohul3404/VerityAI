const App = {
  toastInstance: null,

  async request(url, options = {}) {
    const response = await fetch(url, options);
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.error || "Request failed.");
    return payload;
  },

  percent(value) {
    return `${Math.round(Number(value || 0) * 100)}%`;
  },

  showToast(message, type = "success") {
    const element = document.querySelector("#appToast");
    document.querySelector("#toastMessage").textContent = message;
    element.style.borderLeft = `4px solid ${type === "error" ? "var(--red)" : "var(--green)"}`;
    this.toastInstance = bootstrap.Toast.getOrCreateInstance(element, { delay: 3600 });
    this.toastInstance.show();
  },

  loading(show, message = "Working...") {
    const overlay = document.querySelector("#loadingOverlay");
    document.querySelector("#loadingMessage").textContent = message;
    overlay.classList.toggle("show", show);
    overlay.setAttribute("aria-hidden", String(!show));
  },

  renderIcons() {
    if (window.lucide) lucide.createIcons();
  },
};

document.addEventListener("DOMContentLoaded", async () => {
  const root = document.documentElement;
  const savedTheme = localStorage.getItem("verity-theme") || "light";
  root.setAttribute("data-bs-theme", savedTheme);
  document.querySelector("#themeToggle")?.addEventListener("click", () => {
    const next = root.getAttribute("data-bs-theme") === "dark" ? "light" : "dark";
    root.setAttribute("data-bs-theme", next);
    localStorage.setItem("verity-theme", next);
  });
  document.querySelector("#sidebarToggle")?.addEventListener("click", () => {
    document.querySelector("#sidebar").classList.toggle("open");
  });
  document.addEventListener("click", (event) => {
    const sidebar = document.querySelector("#sidebar");
    if (window.innerWidth < 992 && sidebar.classList.contains("open") && !sidebar.contains(event.target) && !event.target.closest("#sidebarToggle")) {
      sidebar.classList.remove("open");
    }
  });

  try {
    const status = await App.request("/api/status");
    const model = status.metrics.selected_model || "Model ready";
    document.querySelector("#sidebarModel").textContent = model;
    document.dispatchEvent(new CustomEvent("modelStatus", { detail: status }));
  } catch (error) {
    document.querySelector("#sidebarModel").textContent = "Setup required";
    App.showToast(error.message, "error");
  }
  App.renderIcons();
});
