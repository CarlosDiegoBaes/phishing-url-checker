/**
 * Send the pasted URL to POST /check and render the response.
 * Every value is written with textContent, so nothing from the URL is parsed as HTML.
 */
(() => {
  "use strict";

  const SVG_NS = "http://www.w3.org/2000/svg";

  const form = document.getElementById("check-form");
  const input = document.getElementById("url");
  const button = document.getElementById("submit");
  const errorBox = document.getElementById("error");
  const result = document.getElementById("result");

  const BAND_CLASS = {
    "LIKELY SAFE": "safe",
    "SUSPICIOUS": "suspicious",
    "LIKELY PHISHING": "phishing",
  };

  /** Features that are 0 or 1; a 1 is highlighted in the table. */
  const BINARY_FEATURES = new Set(["has_ip_host", "risky_tld", "is_shortener"]);

  /** Show an error message and hide any previous result. */
  function showError(message) {
    errorBox.textContent = message;
    errorBox.hidden = false;
    result.hidden = true;
  }

  /** Format a feature value: whole numbers stay whole, fractions get two decimals. */
  function formatValue(value) {
    return Number.isInteger(value) ? String(value) : value.toFixed(2);
  }

  /** A small inline icon built with DOM calls: a warning triangle or a check mark. */
  function icon(kind) {
    const svg = document.createElementNS(SVG_NS, "svg");
    svg.setAttribute("viewBox", "0 0 24 24");
    svg.setAttribute("aria-hidden", "true");
    svg.setAttribute("focusable", "false");
    const path = document.createElementNS(SVG_NS, "path");
    path.setAttribute("fill", "none");
    path.setAttribute("stroke", "currentColor");
    path.setAttribute("stroke-width", "2");
    path.setAttribute("stroke-linecap", "round");
    path.setAttribute("stroke-linejoin", "round");
    path.setAttribute(
      "d",
      kind === "warn"
        ? "M12 3 2 20h20L12 3Zm0 6v5m0 3v.01"
        : "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18Zm-4 9 3 3 5-6",
    );
    svg.appendChild(path);
    return svg;
  }

  /** Build a table row: a label with an optional description, and a value pill. */
  function row(label, description, value, pillClass) {
    const tr = document.createElement("tr");
    const name = document.createElement("td");
    const title = document.createElement("span");
    title.className = "feature-label";
    title.textContent = label;
    name.appendChild(title);
    if (description) {
      const desc = document.createElement("span");
      desc.className = "feature-desc";
      desc.textContent = description;
      name.appendChild(desc);
    }
    const num = document.createElement("td");
    num.className = "num";
    const pill = document.createElement("span");
    pill.className = "value-pill" + (pillClass ? " " + pillClass : "");
    pill.textContent = value;
    num.appendChild(pill);
    tr.append(name, num);
    return tr;
  }

  /** Draw the score meter and its threshold marker, keeping the label on screen. */
  function renderMeter(score, threshold) {
    const meter = document.getElementById("meter");
    meter.setAttribute("aria-valuenow", score.toFixed(2));
    meter.setAttribute(
      "aria-valuetext",
      `${score.toFixed(2)}, decision threshold ${threshold.toFixed(2)}`,
    );
    document.getElementById("meter-fill").style.width = `${(score * 100).toFixed(1)}%`;
    const marker = document.getElementById("meter-threshold");
    marker.style.left = `${(threshold * 100).toFixed(1)}%`;
    const shift = threshold < 0.12 ? "-10%" : threshold > 0.88 ? "-90%" : "-50%";
    marker.style.setProperty("--label-shift", shift);
    document.getElementById("meter-threshold-label").textContent =
      `threshold ${threshold.toFixed(2)}`;
  }

  /** Fill the warning-rule list and its count badge. */
  function renderFlags(ruleFlags) {
    const flags = document.getElementById("flags");
    const count = document.getElementById("flags-count");
    flags.replaceChildren();
    count.textContent = ruleFlags.length === 1 ? "1 warning" : `${ruleFlags.length} warnings`;
    count.classList.toggle("has-flags", ruleFlags.length > 0);
    if (ruleFlags.length === 0) {
      const li = document.createElement("li");
      li.className = "none";
      const text = document.createElement("span");
      text.textContent = "No warning rules fired.";
      li.append(icon("ok"), text);
      flags.appendChild(li);
      return;
    }
    for (const flag of ruleFlags) {
      const li = document.createElement("li");
      const text = document.createElement("span");
      text.textContent = flag;
      li.append(icon("warn"), text);
      flags.appendChild(li);
    }
  }

  /** Fill the page from a /check response. */
  function render(data) {
    const verdict = document.getElementById("verdict");
    verdict.className = "card verdict " + (BAND_CLASS[data.band] || "");
    document.getElementById("band").textContent = data.band;
    document.getElementById("score").textContent = data.score.toFixed(2);
    renderMeter(data.score, data.threshold);

    const changed = data.url !== data.input;
    document.getElementById("checked-input-row").hidden = !changed;
    document.getElementById("checked-input").textContent = changed ? data.input : "";
    document.getElementById("checked-url").textContent = data.url;
    document.getElementById("explanation").textContent = data.explanation;

    renderFlags(data.rule_flags);

    const contribCard = document.getElementById("contributions-card");
    const contribBody = document.getElementById("contributions");
    contribBody.replaceChildren();
    contribCard.hidden = data.contributions === null;
    for (const item of data.contributions || []) {
      const sign = item.value >= 0 ? "+" : "";
      const tone = item.value > 0 ? "positive" : item.value < 0 ? "negative" : "";
      contribBody.appendChild(row(item.name, "", sign + item.value.toFixed(2), tone));
    }

    document.getElementById("features-note").textContent =
      data.contributions === null
        ? `These are the exact inputs the ${data.model} model scored. It does not report ` +
          "how much each one mattered, so the list is in a fixed order, not ranked by importance."
        : `These are the exact inputs the ${data.model} model scored.`;
    const features = document.getElementById("features");
    features.replaceChildren();
    for (const feature of data.features) {
      const on = BINARY_FEATURES.has(feature.name) && feature.value === 1;
      features.appendChild(
        row(feature.label, feature.description, formatValue(feature.value), on ? "on" : ""),
      );
    }

    errorBox.hidden = true;
    result.hidden = false;
  }

  /** Turn a FastAPI error body into one readable sentence. */
  function errorMessage(body, status) {
    if (body && typeof body.detail === "string") return body.detail;
    if (body && Array.isArray(body.detail) && body.detail.length > 0) {
      return "Please paste a URL (up to 2048 characters).";
    }
    return `The server returned an error (${status}).`;
  }

  /** Show which model and threshold the server loaded, if /healthz answers. */
  async function loadModelInfo() {
    try {
      const response = await fetch("/healthz");
      if (!response.ok) return;
      const body = await response.json();
      const info = document.getElementById("model-info");
      info.textContent = `${body.model} · threshold ${Number(body.threshold).toFixed(2)}`;
      info.hidden = false;
    } catch (err) {
      // The badge is optional; the checker still works without it.
    }
  }

  /** Disable the button and announce that a check is running. */
  function setBusy(busy) {
    button.disabled = busy;
    button.textContent = busy ? "Checking…" : "Check URL";
    form.setAttribute("aria-busy", String(busy));
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const url = input.value.trim();
    if (!url) {
      showError("Please paste a URL first.");
      input.focus();
      return;
    }
    setBusy(true);
    try {
      const response = await fetch("/check", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      });
      const body = await response.json().catch(() => null);
      if (!response.ok) {
        showError(errorMessage(body, response.status));
        return;
      }
      render(body);
    } catch (err) {
      showError("Could not reach the checker. Is the server running?");
    } finally {
      setBusy(false);
    }
  });

  for (const chip of document.querySelectorAll(".chip[data-url]")) {
    chip.addEventListener("click", () => {
      input.value = chip.dataset.url;
      form.requestSubmit();
    });
  }

  loadModelInfo();
})();
