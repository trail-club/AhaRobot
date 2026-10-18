"use strict";

const STATUS_LABELS = {
  pending: "未実行",
  bypassed: "省略",
  running: "実行中",
  succeeded: "成功",
  failed: "失敗",
  unsupported: "未実装",
  skipped: "スキップ",
  cancelled: "中断",
};
const STATE_LABELS = {
  preparing: "準備中",
  idle: "待機中",
  running: "実行中",
  paused: "失敗で停止中",
  completed: "完了",
  failed: "失敗",
  aborted: "中断",
  error: "エラー",
};
const PHASE_LABELS = { restoring: "状態を復元中", executing: "実行中" };
const POLICY_LABELS = {
  pause: "停止して選ぶ",
  retry: "自動で再試行",
  skip: "スキップして続行",
  abort: "中断する",
};
const STARTABLE = ["idle", "completed", "failed", "aborted"];
const RESOLVED = ["succeeded", "unsupported", "skipped"];
const SUMMARY_ORDER = ["succeeded", "failed", "unsupported", "skipped", "cancelled", "bypassed"];

const $ = (id) => document.getElementById(id);
let status = null;
let online = false;
let connected = false;
let selection = null; // ids chosen in this browser
let anchor = null; // last clicked index, for Shift+click ranges
let lastCurrent = null;
let flash = null; // response to the last command

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === "class") node.className = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (value === true) node.setAttribute(key, "");
    else if (value !== false && value != null) node.setAttribute(key, value);
  }
  for (const child of children.flat()) {
    if (child == null || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function formatClock(seconds) {
  if (seconds == null || !Number.isFinite(seconds)) return "--:--";
  const s = Math.max(0, Math.floor(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

function formatValue(value) {
  return typeof value === "string" ? value : JSON.stringify(value);
}

function busy() {
  return status != null && (status.state === "running" || status.state === "paused");
}

async function send(payload) {
  try {
    const response = await fetch("api/command", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const body = await response.json();
    flash = { text: body.message || "", error: !body.accepted };
  } catch (error) {
    flash = { text: `送信できません: ${error}`, error: true };
  }
  const shown = flash;
  setTimeout(() => {
    if (flash === shown) {
      flash = null;
      render();
    }
  }, 6000);
  render();
}

function orderedSelection() {
  return status.subgoals.map((s) => s.id).filter((id) => selection.has(id));
}

function setSelection(ids) {
  selection = new Set(ids);
  flash = null;
  render();
}

function toggle(index, checked, shift) {
  const ids = status.subgoals.map((s) => s.id);
  const [from, to] = shift && anchor != null ? [Math.min(anchor, index), Math.max(anchor, index)] : [index, index];
  for (const id of ids.slice(from, to + 1)) {
    if (checked) selection.add(id);
    else selection.delete(id);
  }
  anchor = index;
  flash = null;
  render();
}

function renderHeader() {
  const task = status ? status.task : null;
  $("task-title").textContent = task ? task.title : "Task Monitor";
  document.title = task ? `${task.title} – Task Monitor` : "Task Monitor";
  const meta = [];
  if (task && task.rulebook) {
    const r = task.rulebook;
    meta.push([r.name, r.version, r.section && `§${r.section}`, r.revision && `(${r.revision})`].filter(Boolean).join(" "));
  }
  if (task && task.time_limit) meta.push(`制限時間 ${formatClock(task.time_limit)}`);
  $("task-meta").textContent = meta.join(" · ");
  const state = status ? status.state : connected ? "waiting" : "offline";
  const badge = $("state");
  badge.className = `badge b-${state}`;
  badge.textContent = STATE_LABELS[state] || (connected ? "runner待ち" : "未接続");
  const indicator = $("online");
  indicator.classList.toggle("on", connected && online);
  indicator.textContent = connected ? (online ? "runner接続" : "runner停止") : "GUI未接続";
}

function renderControls() {
  const policy = $("policy");
  if (!policy.options.length) {
    for (const name of status.policies) policy.append(el("option", { value: name }, POLICY_LABELS[name] || name));
    policy.value = status.defaults.on_failure;
  }
  const startable = online && STARTABLE.includes(status.state);
  $("start").disabled = !startable || selection.size === 0;
  $("select-all").disabled = busy();
  $("select-none").disabled = busy();
  policy.disabled = busy();
  const points = status.subgoals.filter((s) => selection.has(s.id)).reduce((sum, s) => sum + s.points, 0);
  $("selection-count").textContent = `${selection.size} / ${status.subgoals.length} 件選択（最大 ${points} pt）`;

  const active = $("active");
  active.hidden = !busy();
  active.classList.toggle("paused", status.state === "paused");
  const current = status.subgoals.find((s) => s.id === status.current);
  const phase = status.state === "paused" ? "直前の状態へ戻して待機中" : PHASE_LABELS[status.phase] || "";
  $("current").textContent = current ? `${current.title}（${current.id}）${phase ? " — " + phase : ""}` : "";
  for (const id of ["retry", "rewind", "skip"]) $(id).disabled = !online || status.state !== "paused";
  $("abort").disabled = !online || !busy();

  const message = $("message");
  const text = flash ? flash.text : status.message;
  message.textContent = text || "";
  message.classList.toggle("error", Boolean(flash ? flash.error : ["failed", "error", "paused"].includes(status.state)));

  const chosen = status.subgoals.filter((s) => s.selected);
  const done = chosen.filter((s) => RESOLVED.includes(s.status)).length;
  $("bar").style.width = chosen.length && status.run ? `${(100 * done) / chosen.length}%` : "0";
  const counts = {};
  for (const s of status.subgoals) counts[s.status] = (counts[s.status] || 0) + 1;
  $("summary").replaceChildren(
    ...SUMMARY_ORDER.filter((name) => counts[name]).map((name) =>
      el("span", { class: `chip s-${name}` }, `${STATUS_LABELS[name]} ${counts[name]}`),
    ),
  );
}

function renderSubgoals() {
  const groups = [];
  status.subgoals.forEach((s, index) => {
    let group = groups[groups.length - 1];
    if (!group || group.name !== s.group) groups.push((group = { name: s.group, items: [] }));
    group.items.push({ s, index });
  });
  const locked = busy();
  const currentIndex = status.subgoals.findIndex((s) => s.id === status.current);
  // While a run is active the runner's selection is shown; otherwise the browser's.
  const isSelected = (s) => (locked ? s.selected : selection.has(s.id));

  const content = groups.map(({ name, items }) => {
    const chosen = items.filter(({ s }) => isSelected(s)).length;
    const box = el("input", {
      type: "checkbox",
      disabled: locked,
      "aria-label": `${name || "グループ"}を選択`,
      onchange: (event) => {
        for (const { s } of items) {
          if (event.target.checked) selection.add(s.id);
          else selection.delete(s.id);
        }
        render();
      },
    });
    box.checked = chosen === items.length;
    box.indeterminate = chosen > 0 && chosen < items.length;
    const ok = items.filter(({ s }) => s.status === "succeeded").length;
    const head = el("div", { class: "group-head" }, box, el("h2", {}, name || "サブゴール"), el("span", { class: "muted small" }, `成功 ${ok} / ${items.length}`));
    return el("div", { class: "group" }, head, ...items.map(({ s, index }) => renderRow(s, index, locked, currentIndex, isSelected(s))));
  });
  $("subgoals").replaceChildren(...content);

  if (status.current && status.current !== lastCurrent) {
    const row = document.querySelector(`[data-id="${CSS.escape(status.current)}"]`);
    if (row) row.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }
  lastCurrent = status.current;
}

function renderRow(s, index, locked, currentIndex, chosen) {
  const box = el("input", { type: "checkbox", disabled: locked, "aria-label": `${s.title}を選択` });
  box.checked = chosen;
  box.addEventListener("click", (event) => toggle(index, box.checked, event.shiftKey));
  const phase = s.id === status.current && status.phase ? PHASE_LABELS[status.phase] : null;
  const chip = el("span", { class: `chip s-${s.status}` }, phase || STATUS_LABELS[s.status] || s.status);

  const detail = el(
    "div",
    { class: "detail" },
    el("code", {}, s.id),
    " · ",
    s.skill,
    s.available ? null : el("span", { class: "tag na" }, "未実装"),
    s.note ? ` · ${s.note}` : null,
  );
  const body = el("div", { class: "body" }, el("div", { class: "title" }, s.title), detail);
  if (s.message && s.status !== "bypassed") {
    const failed = ["failed", "skipped", "cancelled"].includes(s.status);
    body.append(el("div", { class: "detail" }, el("span", { class: failed ? "fail" : "" }, s.message)));
  }
  const meta = el(
    "div",
    { class: "meta" },
    s.attempts > 1 ? el("span", { class: "tag" }, `試行 ${s.attempts}`) : null,
    s.duration != null ? el("span", { class: "tag" }, `${s.duration.toFixed(1)} s`) : null,
    s.points ? el("span", { class: "tag" }, `${s.points} pt`) : null,
  );

  const actions = el("div", { class: "actions" });
  if (!locked) {
    actions.append(
      el("button", { class: "link", onclick: () => setSelection([s.id]) }, "これだけ"),
      el("button", { class: "link", onclick: () => setSelection(status.subgoals.slice(index).map((x) => x.id)) }, "ここから最後まで"),
      el("button", { class: "link", onclick: () => setSelection(status.subgoals.slice(0, index + 1).map((x) => x.id)) }, "最初からここまで"),
    );
  } else if (status.state === "paused" && index < currentIndex && s.status === "succeeded") {
    actions.append(el("button", { class: "link", onclick: () => send({ command: "rewind", target: s.id }) }, "ここへ戻って再開"));
  }

  const classes = ["subgoal"];
  if (s.id === status.current) classes.push("is-current");
  if (!chosen) classes.push("unselected");
  return el("div", { class: classes.join(" "), "data-id": s.id }, box, chip, body, meta, actions.childElementCount ? actions : null);
}

function renderState() {
  const rows = (object) =>
    Object.entries(object || {}).map(([key, value]) => el("tr", {}, el("td", {}, key), el("td", {}, formatValue(value))));
  const memory = rows(status.memory);
  $("memory").replaceChildren(...(memory.length ? memory : [el("tr", {}, el("td", { colspan: 2 }, "（空）"))]));
  const realized = rows(status.realized);
  $("realized").replaceChildren(...(realized.length ? realized : [el("tr", {}, el("td", { colspan: 2 }, "（不明）"))]));
  const entries = status.log.slice(-100).reverse();
  $("log").replaceChildren(
    ...entries.map((entry) =>
      el("li", { class: entry.level }, el("time", {}, new Date(entry.time * 1000).toLocaleTimeString()), entry.message),
    ),
  );
}

function renderClock() {
  const clock = $("clock");
  if (!status || !status.run) {
    clock.textContent = status && status.task.time_limit ? `0:00 / ${formatClock(status.task.time_limit)}` : "--:--";
    return;
  }
  const end = status.run.finished_at || Date.now() / 1000;
  const elapsed = end - status.run.started_at;
  const limit = status.task.time_limit;
  clock.textContent = limit ? `${formatClock(elapsed)} / ${formatClock(limit)}` : formatClock(elapsed);
  clock.classList.toggle("over", Boolean(limit && elapsed > limit));
}

function render() {
  renderHeader();
  renderClock();
  if (!status) {
    $("message").textContent = connected ? "task_runnerの状態を待っています" : "GUIサーバーに接続しています";
    return;
  }
  if (selection == null) {
    selection = new Set(status.run ? status.run.selection : status.defaults.selection);
  }
  renderControls();
  renderSubgoals();
  renderState();
}

function connect() {
  const events = new EventSource("api/events");
  events.onopen = () => {
    connected = true;
    render();
  };
  events.onmessage = (event) => {
    const payload = JSON.parse(event.data);
    online = payload.online;
    status = payload.status;
    connected = true;
    render();
  };
  events.onerror = () => {
    connected = false;
    render();
  };
}

$("start").addEventListener("click", () =>
  send({ command: "start", subgoals: orderedSelection(), on_failure: $("policy").value }),
);
$("select-all").addEventListener("click", () => setSelection(status.subgoals.map((s) => s.id)));
$("select-none").addEventListener("click", () => setSelection([]));
$("retry").addEventListener("click", () => send({ command: "retry" }));
$("rewind").addEventListener("click", () => send({ command: "rewind" }));
$("skip").addEventListener("click", () => send({ command: "skip" }));
$("abort").addEventListener("click", () => {
  if (window.confirm("実行中のサブゴールを中断しますか？")) send({ command: "abort" });
});
setInterval(renderClock, 500);
render();
connect();
