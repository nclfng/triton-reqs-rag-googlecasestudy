// Triton Requirements Guide - plain fetch calls, no framework.
const form = document.getElementById("ask-form");
const questionBox = document.getElementById("question");
const answerBox = document.getElementById("answer");
const citationList = document.getElementById("citations");
const button = document.getElementById("look-up");

// The answer gets a navy rule on its left only once there is real text in it.
function showAnswer(html, filled) {
  answerBox.innerHTML = html;
  answerBox.classList.toggle("filled", filled);
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

// Turn plain answer text into paragraphs, with [1] style citations in mono.
function renderAnswer(text) {
  return text.replace(/\*\*/g, "").split(/\n\s*\n|\n/).filter(Boolean)
    .map(p => "<p>" + escapeHtml(p).replace(/\[([\d,\s]+)\]/g, '<span class="cite">[$1]</span>') + "</p>")
    .join("");
}

function renderCitations(citations) {
  if (!citations.length) {
    citationList.innerHTML = '<li class="note">This answer cites no pages.</li>';
    return;
  }
  citationList.innerHTML = citations.map(c => `
    <li>
      <span class="num">[${c.n}]</span> ${escapeHtml(c.title)}
      <a href="${escapeHtml(c.url)}" target="_blank" rel="noopener">${escapeHtml(c.url)}</a>
      <p class="excerpt">${escapeHtml(c.excerpt)}</p>
    </li>`).join("");
}

function showRemaining(remaining) {
  document.getElementById("remaining").textContent = remaining === null ? "no cap" : remaining;
}

// Sends the question and calls onText with the answer so far, as it is written.
async function ask(question, onText) {
  const headers = { "Content-Type": "application/json" };
  const ownKey = document.getElementById("own-key").value.trim();
  if (ownKey) headers["X-Gemini-Key"] = ownKey;  // sent for this request only
  const response = await fetch("/api/ask-stream", {
    method: "POST", headers, body: JSON.stringify({ question }),
  });
  if (!response.ok) throw new Error((await response.json()).message || "Something went wrong.");

  // The reply is one JSON object per line: {"delta": "..."} ... then {"done": {...}}.
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "", text = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop();  // keep any half-received line
    for (const line of lines.filter(Boolean)) {
      const event = JSON.parse(line);
      if (event.error) throw new Error(event.error);
      if (event.delta) { text += event.delta; onText(text); }
      if (event.done) return event.done;
    }
  }
  throw new Error("The answer was cut off. Please try again.");
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionBox.value.trim();
  if (!question) return;
  button.disabled = true;
  showAnswer('<p class="note">Searching the catalog</p>', false);
  citationList.innerHTML = '<li class="note">None yet.</li>';
  try {
    const data = await ask(question, text => showAnswer(renderAnswer(text), true));
    showAnswer(renderAnswer(data.answer), true);
    renderCitations(data.citations);
    showRemaining(data.remaining);
    document.getElementById("model").textContent = data.model;
  } catch (error) {
    showAnswer('<p class="error">' + escapeHtml(error.message) + "</p>", false);
    if (/used today/.test(error.message)) showRemaining(0);
    if (/own free Gemini key/.test(error.message)) askForOwnKey();
  }
  button.disabled = false;
});

// Example links fill the question box (the student still presses Look up).
document.querySelectorAll(".examples a").forEach(link => {
  link.addEventListener("click", (event) => {
    event.preventDefault();
    questionBox.value = link.dataset.q;
    questionBox.dispatchEvent(new Event("input"));
    questionBox.focus();
  });
});

questionBox.addEventListener("input", () => {
  document.getElementById("count").textContent = questionBox.value.length + " / 400";
});

// Out of free quota: open the sidebar (collapsed on phones) and point at the key field.
function askForOwnKey() {
  document.getElementById("sidebar-details").setAttribute("open", "");
  document.getElementById("own-key").focus();
}

async function loadSources() {
  const pages = await (await fetch("/api/sources")).json();
  document.getElementById("sources").innerHTML = pages.map(p =>
    `<li>${escapeHtml(p.title)}<span class="date">${escapeHtml(p.scraped)}</span></li>`).join("");
}

async function loadConfig() {
  const config = await (await fetch("/api/config")).json();
  document.getElementById("updated").textContent = config.last_updated;
  if (config.feedback_url) {
    document.getElementById("feedback-link").href = config.feedback_url;
    document.getElementById("feedback").hidden = false;
  }
}

// Draggable column dividers. A divider can also be moved with the arrow keys; double-click resets it.
const page = document.getElementById("page");

function makeResizable(handle, variable, fromLeft, min, defaultWidth) {
  const other = fromLeft ? "--right-w" : "--side-w";
  function setWidth(width) {
    const pageWidth = page.getBoundingClientRect().width;
    const otherWidth = parseInt(getComputedStyle(page).getPropertyValue(other)) || 0;
    const max = Math.min(520, pageWidth - otherWidth - 360);  // always leave room for the main column
    const clamped = Math.max(min, Math.min(max, width));
    page.style.setProperty(variable, clamped + "px");
    try { localStorage.setItem(variable, clamped); } catch (e) {}
  }
  handle.addEventListener("pointerdown", (event) => {
    handle.setPointerCapture(event.pointerId);
    handle.classList.add("dragging");
  });
  handle.addEventListener("pointermove", (event) => {
    if (!handle.classList.contains("dragging")) return;
    const box = page.getBoundingClientRect();
    setWidth(fromLeft ? event.clientX - box.left : box.right - event.clientX);
  });
  handle.addEventListener("pointerup", () => handle.classList.remove("dragging"));
  handle.addEventListener("dblclick", () => setWidth(defaultWidth));
  handle.addEventListener("keydown", (event) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    const now = parseInt(getComputedStyle(page).getPropertyValue(variable));
    const step = (event.key === "ArrowRight" ? 1 : -1) * (fromLeft ? 1 : -1) * 20;
    setWidth(now + step);
  });
  try {  // restore the width from last visit, if any
    const saved = localStorage.getItem(variable);
    if (saved) page.style.setProperty(variable, saved + "px");
  } catch (e) {}
}
makeResizable(document.getElementById("resize-side"), "--side-w", true, 200, 260);
makeResizable(document.getElementById("resize-right"), "--right-w", false, 240, 320);

// On narrow screens start with the sidebar collapsed.
if (window.matchMedia("(max-width: 720px)").matches) {
  document.getElementById("sidebar-details").removeAttribute("open");
}
loadSources();
loadConfig();
