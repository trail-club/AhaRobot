"use strict";

// The robot's touch screen: the state comes from api/events (SSE), touches go
// to api/touch. See aha_tasks/display.py. Embedded in the task GUI
// (display/?embedded), the GUI passes the state from its own SSE stream by
// postMessage, so that a GUI tab holds one SSE connection.

const $ = (id) => document.getElementById(id);
let state = { question: null, captions: [], image: null };
let connected = false;
let pressed = null; // {question, index} while a touch is being sent
let shown = null; // what the stage shows, to keep the image from reloading
let toastTimer = null;

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (value === true) node.setAttribute(key, "");
    else if (value !== false && value != null) node.setAttribute(key, value);
  }
  node.append(...children);
  return node;
}

function toast(text, error = false) {
  const box = $("toast");
  box.textContent = text;
  box.classList.toggle("error", error);
  box.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (box.hidden = true), 2000);
}

async function touch(question, index) {
  if (pressed || !connected) return;
  pressed = { question, index };
  render();
  let reply;
  try {
    const response = await fetch("api/touch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, choice: index }),
    });
    reply = await response.json();
  } catch (error) {
    reply = { accepted: false, message: "Not connected" };
  }
  pressed = null;
  toast(reply.accepted ? `✓ ${reply.message}` : reply.message, !reply.accepted);
  render();
}

function renderQuestion(question) {
  const busy = pressed != null || !connected;
  const buttons = question.choices.map((label, index) => {
    const isPressed = pressed && pressed.question === question.id && pressed.index === index;
    return el(
      "button",
      { class: isPressed ? "choice pressed" : "choice", disabled: busy, onclick: () => touch(question.id, index) },
      label,
    );
  });
  return el(
    "section",
    { class: "question", "aria-label": "Question" },
    question.prompt ? el("p", { class: "prompt" }, question.prompt) : "",
    el("div", { class: "choices" }, ...buttons),
  );
}

function renderStage() {
  const key = JSON.stringify([state.question, state.image, pressed, connected]);
  if (key === shown) return;
  shown = key;
  const parts = [];
  if (state.image) {
    parts.push(el("img", { class: "picture", src: `api/image/${state.image.id}`, alt: "What the robot shows" }));
  }
  if (state.question) parts.push(renderQuestion(state.question));
  if (!parts.length) parts.push(el("div", { class: "idle" }, "AhaRobot"));
  const old = $("stage").querySelector("img.picture");
  if (old && state.image && old.getAttribute("src") === parts[0].getAttribute("src")) parts[0] = old;
  $("stage").replaceChildren(...parts);
}

function renderCaptions() {
  const lines = state.captions;
  $("captions").replaceChildren(
    ...lines.map((line, i) => el("li", { class: i === lines.length - 1 ? "latest" : "" }, line.text)),
  );
}

function render() {
  $("offline").hidden = connected;
  renderStage();
  renderCaptions();
}

function listenToParent() {
  window.addEventListener("message", (event) => {
    if (event.origin !== location.origin || event.source !== window.parent) return;
    if (event.data.screen) state = event.data.screen;
    connected = Boolean(event.data.connected);
    render();
  });
}

function connect() {
  const events = new EventSource("api/events");
  events.onopen = () => {
    connected = true;
    render();
  };
  events.onmessage = (event) => {
    state = JSON.parse(event.data);
    connected = true;
    render();
  };
  events.onerror = () => {
    connected = false;
    render();
  };
}

render();
if (new URLSearchParams(location.search).has("embedded") && window.parent !== window) listenToParent();
else connect();
