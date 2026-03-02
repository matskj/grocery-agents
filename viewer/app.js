const boardCanvas = document.getElementById("boardCanvas");
const ctx = boardCanvas.getContext("2d");

const runSelect = document.getElementById("runSelect");
const reloadRunsBtn = document.getElementById("reloadRunsBtn");
const loadRunBtn = document.getElementById("loadRunBtn");
const followLatestCheck = document.getElementById("followLatest");
const liveRefreshCheck = document.getElementById("liveRefresh");

const prevTickBtn = document.getElementById("prevTickBtn");
const playPauseBtn = document.getElementById("playPauseBtn");
const nextTickBtn = document.getElementById("nextTickBtn");
const speedSelect = document.getElementById("speedSelect");
const tickSlider = document.getElementById("tickSlider");

const turnText = document.getElementById("turnText");
const scoreText = document.getElementById("scoreText");
const modeLabel = document.getElementById("modeLabel");
const tickLabel = document.getElementById("tickLabel");
const runLabel = document.getElementById("runLabel");

const finalScoreEl = document.getElementById("finalScore");
const tickCountEl = document.getElementById("tickCount");
const botCountEl = document.getElementById("botCount");
const gridSizeEl = document.getElementById("gridSize");
const blockedBotsEl = document.getElementById("blockedBots");
const stuckBotsEl = document.getElementById("stuckBots");
const queueViolationsEl = document.getElementById("queueViolations");
const laneCongestionEl = document.getElementById("laneCongestion");

const activeOrdersEl = document.getElementById("activeOrders");
const previewOrdersEl = document.getElementById("previewOrders");
const actionListEl = document.getElementById("actionList");

const BOT_COLORS = [
  "#3b82f6",
  "#ef4444",
  "#8b5cf6",
  "#f59e0b",
  "#22c55e",
  "#ec4899",
  "#06b6d4",
  "#f97316",
  "#6366f1",
  "#10b981",
];

const ITEM_COLORS = {
  cheese: "#e8d17f",
  butter: "#f0d96c",
  yogurt: "#d5b4e8",
  milk: "#d8e6f6",
  rice: "#f6e2b8",
  bananas: "#f2dd6a",
  bread: "#d19f68",
  eggs: "#f2e4ba",
  cereal: "#d8c18e",
  tomatoes: "#df8b7b",
};

let runSummaries = [];
let currentRunName = null;
let currentRunData = null;
let tickIndex = 0;
let playing = false;
let playTimer = null;
let pollTimer = null;
let runPollTimer = null;

function speedMs() {
  return Number(speedSelect.value || "220");
}

async function fetchJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} for ${url}`);
  }
  return response.json();
}

function updateRunSelect() {
  const previous = runSelect.value;
  runSelect.innerHTML = "";
  for (const run of runSummaries) {
    const option = document.createElement("option");
    option.value = run.file;
    const score = run.final_score == null ? "running" : `score ${run.final_score}`;
    option.textContent = `${run.file} (${run.mode || "unknown"}, ${score})`;
    runSelect.appendChild(option);
  }
  if (runSummaries.length === 0) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No run logs found";
    runSelect.appendChild(option);
    return;
  }
  if (followLatestCheck.checked) {
    runSelect.value = runSummaries[0].file;
  } else if (previous && runSummaries.some((r) => r.file === previous)) {
    runSelect.value = previous;
  }
}

async function refreshRuns(loadIfChanged = false) {
  try {
    const payload = await fetchJson("/api/runs");
    runSummaries = payload.runs || [];
    updateRunSelect();

    if (!currentRunName && runSummaries.length > 0) {
      await loadRun(runSummaries[0].file, false);
      return;
    }
    if (followLatestCheck.checked && runSummaries.length > 0) {
      const latest = runSummaries[0].file;
      if (currentRunName !== latest || loadIfChanged) {
        await loadRun(latest, true);
      }
    }
  } catch (error) {
    console.error("refreshRuns failed:", error);
  }
}

function aggregateOrders(orders) {
  const active = new Map();
  const pending = new Map();
  for (const order of orders || []) {
    const item = String(order.item_id || "unknown");
    const status = String(order.status || "");
    if (status === "in_progress") {
      active.set(item, (active.get(item) || 0) + 1);
    } else if (status === "pending") {
      pending.set(item, (pending.get(item) || 0) + 1);
    }
  }
  return { active, pending };
}

function renderOrders(orders) {
  const { active, pending } = aggregateOrders(orders);
  activeOrdersEl.innerHTML = "";
  previewOrdersEl.innerHTML = "";

  for (const [item, count] of active.entries()) {
    const li = document.createElement("li");
    li.textContent = `${item} x${count}`;
    activeOrdersEl.appendChild(li);
  }
  for (const [item, count] of pending.entries()) {
    const li = document.createElement("li");
    li.textContent = `${item} x${count}`;
    previewOrdersEl.appendChild(li);
  }
  if (activeOrdersEl.children.length === 0) {
    const li = document.createElement("li");
    li.textContent = "None";
    activeOrdersEl.appendChild(li);
  }
  if (previewOrdersEl.children.length === 0) {
    const li = document.createElement("li");
    li.textContent = "None";
    previewOrdersEl.appendChild(li);
  }
}

function renderActions(actions) {
  actionListEl.innerHTML = "";
  for (const action of actions || []) {
    const li = document.createElement("li");
    if (action.kind === "move") {
      li.textContent = `bot ${action.bot_id}: move (${action.dx}, ${action.dy})`;
    } else if (action.kind === "pick_up") {
      li.textContent = `bot ${action.bot_id}: pick ${action.item_id}`;
    } else if (action.kind === "drop_off") {
      li.textContent = `bot ${action.bot_id}: drop ${action.order_id}`;
    } else {
      li.textContent = `bot ${action.bot_id}: wait`;
    }
    actionListEl.appendChild(li);
  }
  if (actionListEl.children.length === 0) {
    const li = document.createElement("li");
    li.textContent = "No actions";
    actionListEl.appendChild(li);
  }
}

function drawShelf(x, y, cell, color) {
  ctx.fillStyle = "#8f7458";
  ctx.fillRect(x + cell * 0.2, y + cell * 0.12, cell * 0.6, cell * 0.76);
  ctx.fillStyle = "#5a4636";
  ctx.fillRect(x + cell * 0.18, y + cell * 0.1, cell * 0.04, cell * 0.8);
  ctx.fillRect(x + cell * 0.78, y + cell * 0.1, cell * 0.04, cell * 0.8);
  ctx.fillStyle = color;
  ctx.fillRect(x + cell * 0.27, y + cell * 0.2, cell * 0.46, cell * 0.24);
  ctx.fillRect(x + cell * 0.27, y + cell * 0.52, cell * 0.46, cell * 0.24);
}

function drawBoard(tickData) {
  if (!tickData || !tickData.game_state) {
    ctx.clearRect(0, 0, boardCanvas.width, boardCanvas.height);
    return;
  }
  const state = tickData.game_state;
  const team = tickData.team_summary || {};
  const width = state.grid?.width || 0;
  const height = state.grid?.height || 0;
  if (width <= 0 || height <= 0) {
    ctx.clearRect(0, 0, boardCanvas.width, boardCanvas.height);
    return;
  }

  const canvasW = boardCanvas.width;
  const canvasH = boardCanvas.height;
  ctx.clearRect(0, 0, canvasW, canvasH);

  const margin = 24;
  const cell = Math.floor(Math.min((canvasW - margin * 2) / width, (canvasH - margin * 2) / height));
  const boardW = cell * width;
  const boardH = cell * height;
  const ox = Math.floor((canvasW - boardW) / 2);
  const oy = Math.floor((canvasH - boardH) / 2);

  ctx.fillStyle = "#594634";
  ctx.fillRect(ox - 10, oy - 10, boardW + 20, boardH + 20);

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const isAlt = (x + y) % 2 === 0;
      ctx.fillStyle = isAlt ? "#d6d0c4" : "#cdc6b8";
      ctx.fillRect(ox + x * cell, oy + y * cell, cell, cell);
    }
  }

  const queueLanes = Array.isArray(team.queue_lane_cells) ? team.queue_lane_cells : [];
  const queueRing = Array.isArray(team.dropoff_ring_cells) ? team.dropoff_ring_cells : [];
  const conflictHotspots = Array.isArray(team.conflict_hotspots) ? team.conflict_hotspots : [];

  for (const laneCell of queueLanes) {
    if (!Array.isArray(laneCell) || laneCell.length < 2) {
      continue;
    }
    const x = Number(laneCell[0]);
    const y = Number(laneCell[1]);
    ctx.fillStyle = "#3b82f644";
    ctx.fillRect(ox + x * cell, oy + y * cell, cell, cell);
  }

  for (const ringCell of queueRing) {
    if (!Array.isArray(ringCell) || ringCell.length < 2) {
      continue;
    }
    const x = Number(ringCell[0]);
    const y = Number(ringCell[1]);
    ctx.fillStyle = "#f59e0b52";
    ctx.fillRect(ox + x * cell, oy + y * cell, cell, cell);
  }

  for (const wall of state.grid?.walls || []) {
    const [x, y] = wall;
    ctx.fillStyle = "#594737";
    ctx.fillRect(ox + x * cell, oy + y * cell, cell, cell);
  }

  for (const item of state.items || []) {
    const color = ITEM_COLORS[item.kind] || "#d6ca9f";
    drawShelf(ox + item.x * cell, oy + item.y * cell, cell, color);
  }

  for (const tile of state.grid?.drop_off_tiles || []) {
    const [x, y] = tile;
    ctx.fillStyle = "#6bb379";
    ctx.fillRect(ox + x * cell + cell * 0.2, oy + y * cell + cell * 0.2, cell * 0.6, cell * 0.6);
    ctx.fillStyle = "#dff3df";
    ctx.beginPath();
    ctx.arc(ox + x * cell + cell * 0.5, oy + y * cell + cell * 0.5, cell * 0.14, 0, Math.PI * 2);
    ctx.fill();
  }

  const botsByCell = new Map();
  for (const bot of state.bots || []) {
    const key = `${bot.x},${bot.y}`;
    if (!botsByCell.has(key)) {
      botsByCell.set(key, []);
    }
    botsByCell.get(key).push(bot);
  }

  const queueRoles = team.queue_roles || {};
  const queueViolationByBot = team.is_queue_violation_by_bot || {};
  const nearDropoffBlockingByBot = team.near_dropoff_blocking_by_bot || {};
  const yieldByBot = team.yield_applied_by_bot || {};

  for (const [key, bots] of botsByCell.entries()) {
    const [x, y] = key.split(",").map((v) => Number(v));
    const cx = ox + x * cell + cell / 2;
    const cy = oy + y * cell + cell / 2;
    const radius = Math.max(8, cell * 0.32);
    const spread = Math.min(10, cell * 0.2);
    bots.forEach((bot, index) => {
      const angle = (index / Math.max(1, bots.length)) * Math.PI * 2;
      const bx = cx + Math.cos(angle) * spread;
      const by = cy + Math.sin(angle) * spread;
      const color = BOT_COLORS[Number(bot.id) % BOT_COLORS.length] || "#2563eb";
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(bx, by, radius, 0, Math.PI * 2);
      ctx.fill();
      const isViolation = !!queueViolationByBot[String(bot.id)];
      const isBlocking = !!nearDropoffBlockingByBot[String(bot.id)];
      ctx.strokeStyle = isViolation ? "#ef4444" : isBlocking ? "#f59e0b" : "#ffffff";
      ctx.lineWidth = 2;
      ctx.stroke();

      const carrying = (bot.carrying || []).length;
      ctx.fillStyle = "#ffffff";
      ctx.font = `${Math.max(10, Math.floor(cell * 0.33))}px "Trebuchet MS", Verdana, sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(`${bot.id}:${carrying}`, bx, by);

      const role = String(queueRoles[String(bot.id)] || "");
      const roleTag =
        role === "lead_courier"
          ? "L"
          : role === "queue_courier"
            ? "Q"
            : role === "collector"
              ? "C"
              : role === "yield"
                ? "Y"
                : "";
      if (roleTag) {
        ctx.fillStyle = yieldByBot[String(bot.id)] ? "#fde68a" : "#e2e8f0";
        ctx.font = `${Math.max(9, Math.floor(cell * 0.22))}px "Trebuchet MS", Verdana, sans-serif`;
        ctx.fillText(roleTag, bx, by + radius + 10);
      }
    });
  }

  for (const spot of conflictHotspots) {
    const x = Number(spot?.x);
    const y = Number(spot?.y);
    const count = Number(spot?.count || 0);
    if (!Number.isFinite(x) || !Number.isFinite(y) || count <= 1) {
      continue;
    }
    ctx.fillStyle = `rgba(239,68,68,${Math.min(0.15 + count * 0.07, 0.45)})`;
    ctx.fillRect(ox + x * cell, oy + y * cell, cell, cell);
  }

  const failedMoves = Array.isArray(team.failed_move_attempts) ? team.failed_move_attempts : [];
  for (const move of failedMoves) {
    const from = move?.from;
    if (!Array.isArray(from) || from.length < 2) {
      continue;
    }
    const fx = Number(from[0]);
    const fy = Number(from[1]);
    const tx = fx + Number(move?.dx || 0);
    const ty = fy + Number(move?.dy || 0);
    const sx = ox + fx * cell + cell / 2;
    const sy = oy + fy * cell + cell / 2;
    const ex = ox + tx * cell + cell / 2;
    const ey = oy + ty * cell + cell / 2;
    ctx.strokeStyle = "#ef4444";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(sx, sy);
    ctx.lineTo(ex, ey);
    ctx.stroke();
  }
}

function clampTick(index) {
  if (!currentRunData || !Array.isArray(currentRunData.ticks) || currentRunData.ticks.length === 0) {
    return 0;
  }
  return Math.max(0, Math.min(index, currentRunData.ticks.length - 1));
}

function render() {
  if (!currentRunData || !Array.isArray(currentRunData.ticks) || currentRunData.ticks.length === 0) {
    turnText.textContent = "Turn 0/0";
    scoreText.textContent = "Score: 0";
    tickLabel.textContent = "Tick 0";
    modeLabel.textContent = "Mode: -";
    runLabel.textContent = "Run: -";
    finalScoreEl.textContent = "-";
    tickCountEl.textContent = "0";
    botCountEl.textContent = "-";
    gridSizeEl.textContent = "-";
    blockedBotsEl.textContent = "-";
    stuckBotsEl.textContent = "-";
    queueViolationsEl.textContent = "-";
    laneCongestionEl.textContent = "-";
    activeOrdersEl.innerHTML = "";
    previewOrdersEl.innerHTML = "";
    actionListEl.innerHTML = "";
    drawBoard(null);
    return;
  }

  tickIndex = clampTick(tickIndex);
  const tickData = currentRunData.ticks[tickIndex];
  const state = tickData.game_state || {};
  const team = tickData.team_summary || {};
  const score = state.score || 0;
  const totalTicks = currentRunData.tick_count || currentRunData.ticks.length;

  turnText.textContent = `Turn ${tickData.tick}/${Math.max(totalTicks - 1, 0)}`;
  scoreText.textContent = `Score: ${score}`;
  tickLabel.textContent = `Tick ${tickData.tick}`;
  modeLabel.textContent = `Mode: ${tickData.mode || currentRunData.game_mode?.mode || "unknown"}`;
  runLabel.textContent = `Run: ${currentRunData.file || "-"}`;

  tickSlider.max = String(Math.max(0, currentRunData.ticks.length - 1));
  tickSlider.value = String(tickIndex);

  finalScoreEl.textContent =
    currentRunData.game_over?.final_score != null ? String(currentRunData.game_over.final_score) : "running";
  tickCountEl.textContent = String(currentRunData.ticks.length);
  botCountEl.textContent = String((state.bots || []).length);
  gridSizeEl.textContent = `${state.grid?.width || 0}x${state.grid?.height || 0}`;
  blockedBotsEl.textContent = String(team.blocked_bot_count ?? "-");
  stuckBotsEl.textContent = String(team.stuck_bot_count ?? "-");
  queueViolationsEl.textContent = String(team.queue_violation_count ?? "-");
  laneCongestionEl.textContent = String(team.lane_congestion ?? "-");

  renderOrders(state.orders || []);
  renderActions(tickData.actions || []);
  drawBoard(tickData);
}

function stopPlayback() {
  if (playTimer) {
    clearInterval(playTimer);
    playTimer = null;
  }
  playing = false;
  playPauseBtn.textContent = "Play";
}

function startPlayback() {
  if (!currentRunData || !currentRunData.ticks || currentRunData.ticks.length === 0) {
    return;
  }
  stopPlayback();
  playing = true;
  playPauseBtn.textContent = "Pause";
  playTimer = setInterval(() => {
    if (!currentRunData || tickIndex >= currentRunData.ticks.length - 1) {
      stopPlayback();
      return;
    }
    tickIndex += 1;
    render();
  }, speedMs());
}

async function loadRun(runName, preserveTick) {
  if (!runName) {
    return;
  }
  try {
    const previousTick = tickIndex;
    const payload = await fetchJson(`/api/run/${encodeURIComponent(runName)}`);
    currentRunData = payload;
    currentRunName = runName;
    if (preserveTick) {
      tickIndex = clampTick(previousTick);
    } else {
      tickIndex = 0;
    }
    render();
  } catch (error) {
    console.error("loadRun failed:", error);
  }
}

function setupPolling() {
  if (pollTimer) {
    clearInterval(pollTimer);
  }
  if (runPollTimer) {
    clearInterval(runPollTimer);
  }
  pollTimer = setInterval(async () => {
    if (!liveRefreshCheck.checked || !currentRunName) {
      return;
    }
    await loadRun(currentRunName, true);
  }, 1600);

  runPollTimer = setInterval(async () => {
    if (!liveRefreshCheck.checked) {
      return;
    }
    await refreshRuns(false);
  }, 4500);
}

function bindEvents() {
  reloadRunsBtn.addEventListener("click", async () => {
    await refreshRuns(false);
  });

  loadRunBtn.addEventListener("click", async () => {
    const selected = runSelect.value;
    await loadRun(selected, false);
  });

  runSelect.addEventListener("change", () => {
    if (followLatestCheck.checked) {
      followLatestCheck.checked = false;
    }
  });

  followLatestCheck.addEventListener("change", async () => {
    if (followLatestCheck.checked) {
      await refreshRuns(true);
    }
  });

  liveRefreshCheck.addEventListener("change", () => {
    setupPolling();
  });

  prevTickBtn.addEventListener("click", () => {
    tickIndex = clampTick(tickIndex - 1);
    render();
  });

  nextTickBtn.addEventListener("click", () => {
    tickIndex = clampTick(tickIndex + 1);
    render();
  });

  playPauseBtn.addEventListener("click", () => {
    if (playing) {
      stopPlayback();
    } else {
      startPlayback();
    }
  });

  speedSelect.addEventListener("change", () => {
    if (playing) {
      startPlayback();
    }
  });

  tickSlider.addEventListener("input", () => {
    tickIndex = clampTick(Number(tickSlider.value || "0"));
    render();
  });

  window.addEventListener("resize", () => {
    render();
  });
}

async function init() {
  bindEvents();
  await refreshRuns(true);
  setupPolling();
  render();
}

init().catch((error) => {
  console.error("init failed:", error);
});
