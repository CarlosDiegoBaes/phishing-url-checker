/**
 * Send the pasted URL to POST /check and render the response.
 * Every value is written with textContent, so nothing from the URL is parsed as HTML.
 */
(() => {
  "use strict";

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

  /** Build a table row with a label, an optional description, and a numeric cell. */
  function row(label, description, value) {
    const tr = document.createElement("tr");
    const name = document.createElement("td");
    name.textContent = label;
    if (description) {
      const desc = document.createElement("span");
      desc.className = "desc";
      desc.textContent = description;
      name.appendChild(desc);
    }
    const num = document.createElement("td");
    num.className = "num";
    num.textContent = value;
    tr.append(name, num);
    return tr;
  }

  /** Fill the page from a /check response. */
  function render(data) {
    const verdict = document.getElementById("verdict");
    verdict.className = "card verdict " + (BAND_CLASS[data.band] || "");
    document.getElementById("band").textContent = data.band;
    document.getElementById("score").textContent =
      `p(phish) = ${data.score.toFixed(2)} · threshold ${data.threshold.toFixed(2)}`;
    document.getElementById("meter-fill").style.width = `${(data.score * 100).toFixed(1)}%`;
    document.getElementById("meter-threshold").style.left = `${(data.threshold * 100).toFixed(1)}%`;
    document.getElementById("checked-url").textContent =
      data.url === data.input ? data.url : `${data.input}  →  ${data.url}`;
    document.getElementById("explanation").textContent = data.explanation;

    const flags = document.getElementById("flags");
    flags.replaceChildren();
    if (data.rule_flags.length === 0) {
      const li = document.createElement("li");
      li.className = "none";
      li.textContent = "No warning rules fired.";
      flags.appendChild(li);
    }
    for (const flag of data.rule_flags) {
      const li = document.createElement("li");
      li.textContent = flag;
      flags.appendChild(li);
    }

    const contribCard = document.getElementById("contributions-card");
    const contribBody = document.getElementById("contributions");
    contribBody.replaceChildren();
    contribCard.hidden = data.contributions === null;
    for (const item of data.contributions || []) {
      const sign = item.value >= 0 ? "+" : "";
      contribBody.appendChild(row(item.name, "", sign + item.value.toFixed(2)));
    }

    document.getElementById("features-note").textContent =
      data.contributions === null
        ? `These are the exact inputs the ${data.model} model scored. It does not report ` +
          "how much each one mattered, so the list is in a fixed order, not ranked by importance."
        : `These are the exact inputs the ${data.model} model scored.`;
    const features = document.getElementById("features");
    features.replaceChildren();
    for (const feature of data.features) {
      features.appendChild(row(feature.label, feature.description, formatValue(feature.value)));
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

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const url = input.value.trim();
    if (!url) {
      showError("Please paste a URL first.");
      return;
    }
    button.disabled = true;
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
      button.disabled = false;
    }
  });
})();
