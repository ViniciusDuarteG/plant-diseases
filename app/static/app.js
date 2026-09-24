"use strict";

const $ = (id) => document.getElementById(id);
const input = $("photo-input");
const preview = $("preview");
const button = $("analyze-button");
const percent = new Intl.NumberFormat("pt-BR", { style: "percent", maximumFractionDigits: 1 });
let selectedFile = null;
let previewUrl = null;
let modelPresent = false;
let busy = false;
let revision = 0;
let activeRequest = null;

function updateButton() {
  button.disabled = !selectedFile || !modelPresent || busy;
  button.classList.toggle("loading", busy);
  $("button-label").textContent = busy ? "Analisando sua foto…" : "Analisar foto";
  $("button-arrow").textContent = busy ? "" : "↗";
  $("result-panel").setAttribute("aria-busy", String(busy));
}

function showError(message = "") {
  $("error").textContent = message;
  $("error").hidden = !message;
}

function clearPhoto() {
  revision += 1;
  activeRequest?.abort();
  activeRequest = null;
  busy = false;
  selectedFile = null;
  input.value = "";
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  preview.removeAttribute("src");
  preview.hidden = true;
  $("upload-placeholder").hidden = false;
  $("change-photo").hidden = true;
  $("file-details").hidden = true;
  $("result").hidden = true;
  $("result-placeholder").hidden = false;
  showError();
  updateButton();
}

async function selectPhoto(file) {
  if (!file) return;
  clearPhoto();
  const currentRevision = revision;
  if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) {
    showError("Escolha uma foto JPG, PNG ou WebP.");
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    showError("A foto deve ter no máximo 10 MB.");
    return;
  }
  previewUrl = URL.createObjectURL(file);
  preview.src = previewUrl;
  try {
    await preview.decode();
    if (currentRevision !== revision) return;
    if (preview.naturalWidth * preview.naturalHeight > 20_000_000) {
      clearPhoto();
      showError("A foto deve ter no máximo 20 megapixels. Reduza a resolução e tente novamente.");
      return;
    }
    selectedFile = file;
    preview.hidden = false;
    $("upload-placeholder").hidden = true;
    $("change-photo").hidden = false;
    $("file-details").hidden = false;
    $("file-name").textContent = file.name;
    updateButton();
  } catch {
    if (currentRevision !== revision) return;
    clearPhoto();
    showError("Não foi possível abrir essa foto. Escolha outro arquivo.");
  }
}

async function refreshStatus() {
  $("refresh-status").disabled = true;
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    if (!response.ok) throw new Error();
    const status = await response.json();
    modelPresent = status.model_present;
    $("model-status-text").textContent = status.model_loaded ? "Modelo pronto" : modelPresent ? "Modelo encontrado" : "Modelo pendente";
    $("setup-note").hidden = modelPresent;
  } catch {
    modelPresent = false;
    $("model-status-text").textContent = "Servidor indisponível";
    $("setup-note").hidden = true;
    showError("Não foi possível conectar ao servidor. Confira se a aplicação está rodando e clique em Verificar.");
  } finally {
    $("status-dot").classList.toggle("ready", modelPresent);
    $("refresh-status").disabled = false;
    updateButton();
  }
}

function showResult(data) {
  const prediction = data.prediction;
  $("plant-name").textContent = prediction.plant;
  $("condition-name").textContent = prediction.condition;
  $("confidence-value").textContent = percent.format(prediction.confidence);
  const confidence = Math.max(0, Math.min(100, prediction.confidence * 100));
  $("confidence-fill").style.width = `${confidence}%`;
  $("confidence-meter").setAttribute("aria-valuenow", confidence.toFixed(1));
  $("alternatives-list").replaceChildren();
  for (const alternative of data.alternatives) {
    const row = document.createElement("li");
    const name = document.createElement("span");
    const score = document.createElement("span");
    name.textContent = `${alternative.plant} · ${alternative.condition}`;
    score.textContent = percent.format(alternative.confidence);
    row.append(name, score);
    $("alternatives-list").append(row);
  }
  $("result-placeholder").hidden = true;
  $("result").hidden = false;
  $("model-status-text").textContent = "Modelo pronto";
}

input.addEventListener("change", () => selectPhoto(input.files[0]));
$("remove-photo").addEventListener("click", clearPhoto);
$("refresh-status").addEventListener("click", () => { showError(); refreshStatus(); });
const dropzone = $("dropzone");
for (const event of ["dragenter", "dragover"]) {
  dropzone.addEventListener(event, (e) => { e.preventDefault(); dropzone.classList.add("dragging"); });
}
for (const event of ["dragleave", "drop"]) {
  dropzone.addEventListener(event, (e) => { e.preventDefault(); dropzone.classList.remove("dragging"); });
}
dropzone.addEventListener("drop", (e) => {
  if (e.dataTransfer.files.length !== 1) {
    showError("Selecione apenas uma foto por análise.");
    return;
  }
  selectPhoto(e.dataTransfer.files[0]);
});

$("upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!selectedFile || !modelPresent || busy) return;
  const currentRevision = revision;
  busy = true;
  showError();
  $("result").hidden = true;
  $("result-placeholder").hidden = false;
  updateButton();
  const form = new FormData();
  form.append("file", selectedFile);
  activeRequest = new AbortController();
  try {
    const response = await fetch("/api/predict", { method: "POST", body: form, signal: activeRequest.signal });
    const data = await response.json();
    if (currentRevision !== revision) return;
    if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Não foi possível analisar essa foto.");
    showResult(data);
  } catch (error) {
    if (currentRevision === revision && error.name !== "AbortError") {
      showError(error instanceof TypeError ? "A conexão falhou. Confira o servidor e tente novamente." : error.message);
    }
  } finally {
    if (currentRevision === revision) {
      busy = false;
      activeRequest = null;
      updateButton();
    }
  }
});

refreshStatus();
