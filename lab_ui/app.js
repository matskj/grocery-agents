const state = {
  runs: [],
  activeRun: null,
  runMeta: null,
  tick: 0,
  frame: null,
  nextFrame: null,
  playing: false,
  playTimer: null,
  forkSession: null,
};

const els = {
  runSelect: document.getElementById('runSelect'),
  tickInput: document.getElementById('tickInput'),
  prevBtn: document.getElementById('prevBtn'),
  nextBtn: document.getElementById('nextBtn'),
  playBtn: document.getElementById('playBtn'),
  speedInput: document.getElementById('speedInput'),
  loadBtn: document.getElementById('loadBtn'),
  board: document.getElementById('board'),
  boardMeta: document.getElementById('boardMeta'),
  tickDiff: document.getElementById('tickDiff'),
  timeline: document.getElementById('timeline'),
  forkBtn: document.getElementById('forkBtn'),
  forkStepBtn: document.getElementById('forkStepBtn'),
  forkStep10Btn: document.getElementById('forkStep10Btn'),
  forkOut: document.getElementById('forkOut'),
  episodesInput: document.getElementById('episodesInput'),
  policySelect: document.getElementById('policySelect'),
  evalBtn: document.getElementById('evalBtn'),
  evalOut: document.getElementById('evalOut'),
};

async function api(path, options = {}) {
  const res = await fetch(path, options);
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status} ${body}`);
  }
  return res.json();
}

async function init() {
  await loadRuns();
  bind();
}

function bind() {
  els.loadBtn.onclick = () => loadCurrentRun();
  els.prevBtn.onclick = () => jumpTick(state.tick - 1);
  els.nextBtn.onclick = () => jumpTick(state.tick + 1);
  els.playBtn.onclick = togglePlay;
  els.forkBtn.onclick = forkFromTick;
  els.forkStepBtn.onclick = () => forkStep(1, null);
  els.forkStep10Btn.onclick = () => forkStep(10, 'auto');
  els.evalBtn.onclick = runEval;
}

async function loadRuns() {
  const data = await api('/api/runs');
  state.runs = data.runs || [];
  els.runSelect.innerHTML = '';
  for (const run of state.runs) {
    const option = document.createElement('option');
    option.value = run.run_id;
    option.textContent = `${run.file_name} (${run.mode || 'unknown'})`;
    els.runSelect.appendChild(option);
  }
  if (state.runs.length > 0) {
    state.activeRun = state.runs[0].run_id;
    await loadCurrentRun();
  }
}

async function loadCurrentRun() {
  state.activeRun = els.runSelect.value;
  state.runMeta = await api(`/api/run/${encodeURIComponent(state.activeRun)}/meta`);
  state.tick = state.runMeta.first_tick || 0;
  els.tickInput.value = state.tick;
  await loadTick(state.tick);
}

async function loadTick(tick) {
  if (!state.runMeta) return;
  const minTick = state.runMeta.first_tick ?? 0;
  const maxTick = state.runMeta.last_tick ?? minTick;
  const clamped = Math.max(minTick, Math.min(maxTick, tick));
  state.tick = clamped;
  els.tickInput.value = clamped;

  state.frame = await api(`/api/run/${encodeURIComponent(state.activeRun)}/tick/${clamped}`);
  state.nextFrame = clamped < maxTick
    ? await api(`/api/run/${encodeURIComponent(state.activeRun)}/tick/${clamped + 1}`)
    : null;

  render();
}

async function jumpTick(t) {
  await loadTick(t);
}

function togglePlay() {
  if (state.playing) {
    clearInterval(state.playTimer);
    state.playing = false;
    els.playBtn.textContent = 'Play';
    return;
  }

  state.playing = true;
  els.playBtn.textContent = 'Pause';
  const speed = Math.max(50, Number(els.speedInput.value) || 250);
  state.playTimer = setInterval(async () => {
    if (!state.runMeta) return;
    const maxTick = state.runMeta.last_tick ?? state.tick;
    if (state.tick >= maxTick) {
      togglePlay();
      return;
    }
    await jumpTick(state.tick + 1);
  }, speed);
}

function render() {
  renderBoard();
  renderMeta();
  renderDiff();
  renderTimeline();
}

function renderBoard() {
  const frame = state.frame;
  if (!frame || !frame.game_state) return;
  const gs = frame.game_state;
  const { width, height, walls, drop_off_tiles } = gs.grid;
  const canvas = els.board;
  const ctx = canvas.getContext('2d');

  const cell = Math.max(16, Math.floor(Math.min(850 / Math.max(width, 1), 580 / Math.max(height, 1))));
  canvas.width = width * cell;
  canvas.height = height * cell;

  ctx.fillStyle = '#fefbf4';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      ctx.strokeStyle = '#ece4d7';
      ctx.strokeRect(x * cell, y * cell, cell, cell);
    }
  }

  ctx.fillStyle = '#53504b';
  for (const w of walls || []) {
    ctx.fillRect(w[0] * cell, w[1] * cell, cell, cell);
  }

  ctx.fillStyle = '#f97316';
  for (const d of drop_off_tiles || []) {
    ctx.fillRect(d[0] * cell + 2, d[1] * cell + 2, cell - 4, cell - 4);
  }

  ctx.fillStyle = '#0f766e';
  for (const item of gs.items || []) {
    ctx.fillRect(item.x * cell + 4, item.y * cell + 4, cell - 8, cell - 8);
  }

  for (const bot of gs.bots || []) {
    const hue = (Number(bot.id) * 43) % 360;
    ctx.fillStyle = `hsl(${hue} 75% 45%)`;
    ctx.beginPath();
    ctx.arc(bot.x * cell + cell / 2, bot.y * cell + cell / 2, Math.max(5, cell / 3), 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = '#111';
    ctx.font = `${Math.max(10, Math.floor(cell / 3))}px IBM Plex Mono`;
    ctx.fillText(String(bot.id), bot.x * cell + 2, bot.y * cell + cell - 4);
  }
}

function renderMeta() {
  const frame = state.frame;
  if (!frame || !frame.game_state) return;
  const gs = frame.game_state;
  els.boardMeta.textContent = [
    `run=${state.activeRun}`,
    `tick=${gs.tick}`,
    `score=${gs.score}`,
    `active_order_index=${gs.active_order_index}`,
    `bots=${gs.bots.length} items=${gs.items.length} orders=${gs.orders.length}`,
  ].join(' | ');
}

function renderDiff() {
  const a = state.frame?.game_state;
  const b = state.nextFrame?.game_state;
  if (!a || !b) {
    els.tickDiff.textContent = 'No next tick available.';
    return;
  }

  const delta = {
    tick: `${a.tick} -> ${b.tick}`,
    score: `${a.score} -> ${b.score}`,
    bots_count: `${a.bots.length} -> ${b.bots.length}`,
    items_count: `${a.items.length} -> ${b.items.length}`,
    orders_delivered: `${countDelivered(a.orders)} -> ${countDelivered(b.orders)}`,
    actions: state.frame.actions || [],
  };
  els.tickDiff.textContent = JSON.stringify(delta, null, 2);
}

function renderTimeline() {
  const frame = state.frame;
  if (!frame?.game_state) return;
  const lines = [];
  for (const bot of frame.game_state.bots) {
    const action = (frame.actions || []).find(a => a.bot_id === bot.id) || { kind: 'wait' };
    const carrying = bot.carrying?.length ? bot.carrying.join(',') : '-';
    lines.push(`bot=${bot.id} pos=(${bot.x},${bot.y}) carrying=[${carrying}] action=${action.kind}`);
  }
  els.timeline.textContent = lines.join('\n');
}

function countDelivered(orders) {
  return (orders || []).filter(o => o.status === 'delivered').length;
}

async function forkFromTick() {
  if (!state.activeRun) return;
  try {
    const result = await api('/api/sim/fork', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ run_id: state.activeRun, tick: state.tick }),
    });
    state.forkSession = result.session_id;
    els.forkOut.textContent = JSON.stringify(result, null, 2);
  } catch (err) {
    els.forkOut.textContent = `fork failed: ${err.message}`;
  }
}

async function forkStep(steps, policyMode) {
  if (!state.forkSession) {
    els.forkOut.textContent = 'fork a session first';
    return;
  }
  try {
    const payload = { steps };
    if (policyMode) payload.policy_mode = policyMode;
    const result = await api(`/api/sim/${state.forkSession}/step`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const metrics = await api(`/api/sim/${state.forkSession}/metrics`);
    els.forkOut.textContent = JSON.stringify({ step: result, metrics }, null, 2);
  } catch (err) {
    els.forkOut.textContent = `step failed: ${err.message}`;
  }
}

async function runEval() {
  try {
    const result = await api('/api/eval/batch', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        episodes: Number(els.episodesInput.value || 10),
        policy: els.policySelect.value,
      }),
    });
    els.evalOut.textContent = JSON.stringify(result.summary, null, 2);
  } catch (err) {
    els.evalOut.textContent = `eval failed: ${err.message}`;
  }
}

init().catch((err) => {
  console.error(err);
  els.tickDiff.textContent = err.message;
});
