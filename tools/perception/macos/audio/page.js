const $ = (id) => document.getElementById(id);

function remembered(key) {
  try { return localStorage.getItem(key) || ''; } catch { return ''; }
}
function remember(key, value) {
  try { localStorage.setItem(key, value); } catch { /* private window */ }
}

async function post(path, body = {}) {
  const response = await fetch(path, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  });
  if (!response.ok) {
    const { error } = await response.json();
    throw new Error(error);
  }
}

function selectButtons(group, value) {
  for (const button of group.querySelectorAll('button')) {
    button.classList.toggle('active', button.dataset.value === value);
    button.setAttribute('aria-pressed', button.dataset.value === value);
  }
}

function fillButtons(group, names) {
  group.replaceChildren(...Object.entries(names).map(([value, name]) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.dataset.value = value;
    button.textContent = name;
    return button;
  }));
}

function showError(error) {
  $('error').textContent = error;
  $('error').hidden = !error;
}

// The connect card and the status chip, which both pages have.
function renderConnection(state, targets) {
  const { status = 'idle', step, error, target } = state;
  const ready = status === 'ready';
  const busy = status === 'connecting' || status === 'closing';

  $('connect').hidden = ready;
  $('statusText').textContent = {
    idle: '未接続', connecting: '接続しています', closing: '切断しています',
    error: '接続できません', ready: `${targets[target] || target} に接続中`,
  }[status];
  $('statusDot').className = 'dot ' + (ready ? 'ok' : status === 'error' ? 'bad' : busy ? 'busy' : '');

  if (error) showError(error); else if (!busy) showError('');
  $('step').textContent = step || '';
  $('connectButton').hidden = busy;
  $('connectButton').textContent = status === 'error' ? 'もう一度接続する' : '接続する';
  $('cancelButton').hidden = status !== 'connecting';
  for (const control of $('connect').querySelectorAll('button:not(#connectButton):not(#cancelButton), input, select')) {
    control.disabled = busy;
  }
  return ready;
}

function listenEvents(onEvent) {
  const source = new EventSource('/events');
  source.onmessage = ({ data }) => onEvent(JSON.parse(data));
  source.onerror = () => {
    $('statusText').textContent = 'ツールが終了しました';
    $('statusDot').className = 'dot bad';
  };
}

function checkEnglish(hint) {
  const notEnglish = /[^\x00-\x7F‘’“”—–…]/.test($('text').value);
  $('speakHint').classList.toggle('warn', notEnglish);
  $('speakHint').textContent = notEnglish
    ? '読み上げは英語のみです。英語以外の文字は正しく発音できません。'
    : hint;
}

function showHint(error) {
  $('speakHint').textContent = error.message;
  $('speakHint').classList.add('warn');
}

// Buttons and keys that behave the same on both pages.
function bindControls(connection, submit, check) {
  $('connectButton').addEventListener('click', async () => {
    const body = connection();
    remember('user', body.user);
    showError('');
    try {
      await post('/api/connect', body);
    } catch (error) {
      showError(error.message);
    }
  });
  $('cancelButton').addEventListener('click', () => post('/api/disconnect'));
  $('disconnectButton').addEventListener('click', () => post('/api/disconnect'));
  $('stopButton').addEventListener('click', () => post('/api/stop'));
  $('text').addEventListener('input', check);
  $('text').addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      $('speakForm').requestSubmit();
    }
  });
  $('speakForm').addEventListener('submit', async (event) => {
    event.preventDefault();
    const text = $('text').value.trim();
    if (!text || state.activity === 'speaking') return;
    try {
      await submit(text);
    } catch (error) {
      showHint(error);
    }
  });
}

async function loadDefaults() {
  const defaults = await (await fetch('/api/defaults')).json();
  fillButtons($('targets'), defaults.targets);
  $('user').value = defaults.user || remembered('user');
  $('userField').hidden = !defaults.user_required;
  return defaults;
}
