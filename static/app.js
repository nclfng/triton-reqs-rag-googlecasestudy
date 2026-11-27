// Triton Requirements Guide - plain fetch calls, no framework.
const form = document.getElementById("ask-form");
const questionBox = document.getElementById("question");
const answerBox = document.getElementById("answer");
const citationList = document.getElementById("citations");
const button = document.getElementById("look-up");

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
  answerBox.innerHTML = '<p class="note">Searching the catalog</p>';
  citationList.innerHTML = '<li class="note">None yet.</li>';
  try {
    const data = await ask(question, text => { answerBox.innerHTML = renderAnswer(text); });
    answerBox.innerHTML = renderAnswer(data.answer);
    renderCitations(data.citations);
    showRemaining(data.remaining);
    document.getElementById("model").textContent = data.model;
  } catch (error) {
    answerBox.innerHTML = '<p class="error">' + escapeHtml(error.message) + "</p>";
    if (/used today/.test(error.message)) showRemaining(0);
  }
  button.disabled = false;
});

questionBox.addEventListener("input", () => {
  document.getElementById("count").textContent = questionBox.value.length + " / 400";
});

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

// On narrow screens start with the sidebar collapsed.
if (window.matchMedia("(max-width: 720px)").matches) {
  document.getElementById("sidebar-details").removeAttribute("open");
}
loadSources();
loadConfig();
