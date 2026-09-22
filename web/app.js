/**
 * Franka Emika Panda MuJoCo VLA Workspace — Frontend Client
 * ---------------------------------------------------------
 * Features:
 * - Direct MuJoCo Dual-Camera Realtime Streaming (Top + Wrist)
 * - Interactive Natural Language Chat Console for controlling Franka 7-DoF arm with VLA policy
 * - Full Telemetry Readouts (Joint angles q1..q7, Cartesian TCP XYZ, Gripper opening mm)
 * - Diverse Object Roster (Soup Can, Mustard, Blue Box, Red Cylinder, Green Cube, Basket)
 * - VLA Policy Switcher with Ghosted Non-Ready Options (Octo-Small, RT-1, OpenVLA, Pi0)
 */

// ============================================================================
// 1. STATE & DOM REFERENCES
// ============================================================================
const state = {
  activeTarget: 'can',
  ablationMode: 'memory',
  vlaModel: 'MiniVLA (VQ-Libero90)',
  controlParadigm: 'vla', // 'vla', 'vlm_ik', 'offline_fallback'
  envMode: 'libero', // 'libero' (Benchmark) or 'tabletop' (Custom)
  liberoTiers: {},
  liberoCurrentTier: 1,
  liberoCurrentTaskId: 0,
  liberoInstruction: '',
  taskStatus: 'IDLE',
  viewLayout: 'dual', // 'dual', 'top', 'wrist'
  lastFrameTime: performance.now(),
  fps: 0,
  frameCount: 0,
  sseConnected: false,
  remoteApiUrl: '',
  allObjects: {}
};

// DOM Elements - Environment & LIBERO Suite
const btnModeLibero = document.getElementById('btn-mode-libero');
const btnModeTabletop = document.getElementById('btn-mode-tabletop');
const liberoSuitePanel = document.getElementById('libero-suite-panel');
const liberoTierSelect = document.getElementById('libero-tier-select');
const liberoTaskSelect = document.getElementById('libero-task-select');
const liberoDefaultPrompt = document.getElementById('libero-default-prompt');
const btnUseDefaultPrompt = document.getElementById('btn-use-default-prompt');
const btnLiberoExecute = document.getElementById('btn-libero-execute');
const btnLiberoStop = document.getElementById('btn-libero-stop');
const btnLiberoReset = document.getElementById('btn-libero-reset');
const liberoSuiteBadge = document.getElementById('libero-suite-badge');
const camTopLabel = document.getElementById('cam-top-label');
const camWristLabel = document.getElementById('cam-wrist-label');

// DOM Elements - Action Chunks & Progress
const chunkDx = document.getElementById('chunk-dx');
const chunkDy = document.getElementById('chunk-dy');
const chunkDz = document.getElementById('chunk-dz');
const chunkDr = document.getElementById('chunk-dr');
const chunkDp = document.getElementById('chunk-dp');
const chunkDyaw = document.getElementById('chunk-dyaw');
const chunkGrp = document.getElementById('chunk-grp');
const progressBarFill = document.getElementById('progress-bar-fill');
const progressPctVal = document.getElementById('progress-pct-val');
const progressStepVal = document.getElementById('progress-step-val');

// DOM Elements - Headers & Badges
const vlaModelSelect = document.getElementById('vla-model-select');
const btnConfigApi = document.getElementById('btn-config-api');
const btnAblationMemory = document.getElementById('btn-ablation-memory');
const btnAblationRaw = document.getElementById('btn-ablation-raw');
const btnParadigmVla = document.getElementById('btn-paradigm-vla');
const btnParadigmVlmIk = document.getElementById('btn-paradigm-vlm-ik');
const btnParadigmFallback = document.getElementById('btn-paradigm-fallback');
const taskStatusPill = document.getElementById('task-status-pill');
const pillOfflineFallback = document.getElementById('pill-offline-fallback');
const statusDot = document.getElementById('status-dot');
const taskStatusText = document.getElementById('task-status-text');
const fpsValEl = document.getElementById('fps-val');
const chatAgentNameEl = document.getElementById('chat-agent-name');

// DOM Elements - Chat Console
const chatFeed = document.getElementById('chat-feed');
const chatForm = document.getElementById('chat-form');
const chatInput = document.getElementById('chat-input');
const btnChatSend = document.getElementById('btn-chat-send');
const chipsContainer = document.getElementById('chips-container');
const btnQuickRandomize = document.getElementById('btn-quick-randomize');
const btnQuickReset = document.getElementById('btn-quick-reset') || document.getElementById('btn-quick-home');
const btnQuickOpenGripper = document.getElementById('btn-quick-open-gripper');
const btnQuickCloseGripper = document.getElementById('btn-quick-close-gripper');

// DOM Elements - Dual Viewport
const viewportContainer = document.getElementById('viewport-container');
const streamTopFrame = document.getElementById('stream-top-frame');
const streamWristFrame = document.getElementById('stream-wrist-frame');
const btnViewDual = document.getElementById('btn-view-dual');
const btnViewTop = document.getElementById('btn-view-top');
const btnViewWrist = document.getElementById('btn-view-wrist');
const taskResultOverlay = document.getElementById('task-result-overlay');
const btnDismissBanner = document.getElementById('btn-dismiss-banner');
const taskResultDesc = document.getElementById('task-result-desc');

// DOM Elements - Object Roster & Dynamic Prompt
const rosterChipsContainer = document.getElementById('roster-chips-container');
const dpContentText = document.getElementById('dp-content-text');
const dpPhaseTag = document.getElementById('dp-phase-tag');

// DOM Elements - Telemetry Strip
const jointEls = Array.from({ length: 7 }, (_, i) => document.getElementById(`j${i + 1}-val`));
const tcpXEl = document.getElementById('tcp-x');
const tcpYEl = document.getElementById('tcp-y');
const tcpZEl = document.getElementById('tcp-z');
const gripperFill = document.getElementById('gripper-fill');
const gripperMmVal = document.getElementById('gripper-mm-val');
const gripperStateTag = document.getElementById('gripper-state-tag');
const stepCounterVal = document.getElementById('step-counter-val');
const basketContainmentTag = document.getElementById('basket-containment-tag');

// DOM Elements - API Modal
const apiModal = document.getElementById('api-modal');
const inputGeminiKey = document.getElementById('input-gemini-key');
const inputApiUrl = document.getElementById('input-api-url');
const btnCloseModal = document.getElementById('btn-close-modal');
const btnCancelModal = document.getElementById('btn-cancel-modal');
const btnSaveApi = document.getElementById('btn-save-api');


// ============================================================================
// 2. SERVER-SENT EVENTS (SSE) STREAM RECEIVER
// ============================================================================
function connectSSEStream() {
  const evtSource = new EventSource('/stream');

  evtSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      handleStreamPayload(data);
    } catch (err) {
      console.error('[SSE] Failed to parse payload:', err);
    }
  };

  evtSource.onerror = (err) => {
    console.warn('[SSE] Disconnected, reconnecting in 2s...', err);
    statusDot.className = 'status-dot dot-idle';
    taskStatusText.textContent = 'RECONNECTING';
    evtSource.close();
    setTimeout(connectSSEStream, 2000);
  };
}

function handleStreamPayload(data) {
  // 1. Calculate FPS
  state.frameCount++;
  const now = performance.now();
  const delta = now - state.lastFrameTime;
  if (delta >= 1000) {
    state.fps = Math.round((state.frameCount * 1000) / delta);
    fpsValEl.textContent = state.fps;
    state.frameCount = 0;
    state.lastFrameTime = now;
  }

  // 2. Render Camera Frames directly from MuJoCo offscreen renders
  if (data.frame_top_b64 && streamTopFrame) {
    streamTopFrame.src = data.frame_top_b64;
  }
  if (data.frame_wrist_b64 && streamWristFrame) {
    streamWristFrame.src = data.frame_wrist_b64;
  }

  // 3. Update Status Badges
  state.taskStatus = data.status || 'IDLE';
  taskStatusText.textContent = state.taskStatus;
  statusDot.className = 'status-dot dot-' + state.taskStatus.toLowerCase();

  // Sync control paradigm if provided
  if (data.control_paradigm && data.control_paradigm !== state.controlParadigm) {
    state.controlParadigm = data.control_paradigm;
    updateParadigmUI(data.control_paradigm);
  }

  // Offline Fallback / Control Paradigm status pill
  if (pillOfflineFallback) {
    if (data.is_offline_fallback || state.controlParadigm === 'offline_fallback') {
      pillOfflineFallback.style.display = 'inline-flex';
      pillOfflineFallback.textContent = 'OFFLINE FALLBACK';
      pillOfflineFallback.style.background = 'rgba(245, 158, 11, 0.18)';
      pillOfflineFallback.style.color = '#f59e0b';
      pillOfflineFallback.style.border = '1px solid rgba(245, 158, 11, 0.4)';
      pillOfflineFallback.title = 'Hệ thống đang chạy chế độ Offline Fallback';
    } else if (state.controlParadigm === 'vlm_ik') {
      pillOfflineFallback.style.display = 'inline-flex';
      pillOfflineFallback.textContent = 'VLM + IK';
      pillOfflineFallback.style.background = 'rgba(6, 182, 212, 0.18)';
      pillOfflineFallback.style.color = '#06b6d4';
      pillOfflineFallback.style.border = '1px solid rgba(6, 182, 212, 0.4)';
      pillOfflineFallback.title = 'Chế độ VLM Spatial Perception + Differential IK';
    } else if (data.vla_mode_tag && data.vla_mode_tag !== 'OFFLINE FALLBACK') {
      pillOfflineFallback.style.display = 'inline-flex';
      pillOfflineFallback.textContent = data.vla_mode_tag;
      pillOfflineFallback.style.background = 'rgba(16, 185, 129, 0.18)';
      pillOfflineFallback.style.color = '#10b981';
      pillOfflineFallback.style.border = '1px solid rgba(16, 185, 129, 0.4)';
      pillOfflineFallback.title = `Chính sách VLA: ${data.vla_mode_tag}`;
    } else {
      pillOfflineFallback.style.display = 'none';
    }
  }

  // 4. Update Dynamic Prompt & Phase
  if (data.dynamic_prompt) {
    dpContentText.textContent = data.dynamic_prompt;
  }
  const currentPhase = (data.telemetry && data.telemetry.phase) ? data.telemetry.phase.toUpperCase() : 'IDLE';
  dpPhaseTag.textContent = currentPhase;

  // 5. Update Telemetry Strip
  const tel = data.telemetry || {};
  if (Array.isArray(tel.qpos) && tel.qpos.length >= 7) {
    tel.qpos.forEach((rad, idx) => {
      if (jointEls[idx]) {
        const deg = (rad * 180 / Math.PI).toFixed(1);
        jointEls[idx].textContent = `${deg}°`;
      }
    });
  }

  if (Array.isArray(tel.tcp_pos) && tel.tcp_pos.length >= 3) {
    tcpXEl.textContent = Number(tel.tcp_pos[0]).toFixed(3);
    tcpYEl.textContent = Number(tel.tcp_pos[1]).toFixed(3);
    tcpZEl.textContent = Number(tel.tcp_pos[2]).toFixed(3);
  }

  // Gripper
  const grpMm = tel.gripper_width_mm !== undefined ? Number(tel.gripper_width_mm) : 80.0;
  gripperMmVal.textContent = `${grpMm.toFixed(1)} mm`;
  const fillPct = Math.min(100, Math.max(0, (grpMm / 80.0) * 100));
  gripperFill.style.width = `${fillPct}%`;

  if (grpMm < 5.0) {
    gripperStateTag.textContent = 'CLOSED';
    gripperStateTag.style.color = 'var(--text-muted)';
  } else if (grpMm < 65.0) {
    gripperStateTag.textContent = 'GRASPING';
    gripperStateTag.style.color = 'var(--accent-emerald)';
  } else {
    gripperStateTag.textContent = 'OPEN';
    gripperStateTag.style.color = 'var(--accent-cyan)';
  }

  // Action Chunks / 7-DoF Action Vector
  if (tel.last_action && Array.isArray(tel.last_action) && tel.last_action.length >= 7) {
    const a = tel.last_action;
    if (chunkDx) chunkDx.textContent = `dx: ${Number(a[0]).toFixed(3)}`;
    if (chunkDy) chunkDy.textContent = `dy: ${Number(a[1]).toFixed(3)}`;
    if (chunkDz) chunkDz.textContent = `dz: ${Number(a[2]).toFixed(3)}`;
    if (chunkDr) chunkDr.textContent = `dr: ${Number(a[3]).toFixed(2)}`;
    if (chunkDp) chunkDp.textContent = `dp: ${Number(a[4]).toFixed(2)}`;
    if (chunkDyaw) chunkDyaw.textContent = `dy: ${Number(a[5]).toFixed(2)}`;
    if (chunkGrp) {
      const g = Number(a[6]);
      chunkGrp.textContent = `grp: ${g > 0 ? '+1.0 (Close)' : '-1.0 (Open)'}`;
      chunkGrp.style.color = g > 0 ? 'var(--accent-emerald)' : 'var(--accent-cyan)';
    }
  }

  // Task Progress (%)
  if (tel.progress_pct !== undefined) {
    const p = Math.min(100, Math.max(0, Number(tel.progress_pct)));
    if (progressBarFill) progressBarFill.style.width = `${p}%`;
    if (progressPctVal) progressPctVal.textContent = `${p.toFixed(1)}%`;
    if (progressStepVal) progressStepVal.textContent = `Step ${tel.step || 0} / ${tel.max_steps || 150}`;
  }

  // Gripper state tag override if provided from LIBERO
  if (tel.gripper_state) {
    gripperStateTag.textContent = tel.gripper_state;
    gripperStateTag.style.color = tel.gripper_state === 'OPEN' ? 'var(--accent-cyan)' : 'var(--accent-emerald)';
  }

  // Step counter
  const stepIdx = data.step || 0;
  stepCounterVal.textContent = `Step: ${stepIdx} / ${tel.max_steps || 160}`;

  // Environment mode sync
  if (data.env_mode && data.env_mode !== state.envMode) {
    state.envMode = data.env_mode;
    updateEnvModeUI(data.env_mode);
  }

  // Basket containment
  const isContained = tel.metrics && tel.metrics.is_in_basket;
  if (isContained) {
    basketContainmentTag.textContent = 'Trong rỏ';
    basketContainmentTag.style.color = 'var(--accent-emerald)';
  } else {
    basketContainmentTag.textContent = state.envMode === 'libero' ? (tel.is_success ? 'Thành công' : 'Đang thực thi') : 'Ngoài rỏ';
    basketContainmentTag.style.color = tel.is_success ? 'var(--accent-emerald)' : 'var(--text-muted)';
  }

  // Success Banner
  if (state.taskStatus === 'SUCCESS') {
    taskResultOverlay.classList.remove('hidden');
    if (state.envMode === 'libero') {
      taskResultDesc.textContent = `Tác vụ LIBERO '${tel.task_name || 'Benchmark'}' đã hoàn thành xuất sắc sau ${stepIdx} bước!`;
    } else {
      taskResultDesc.textContent = `Vật thể '${state.activeTarget}' đã được gắp và đặt an toàn vào rỏ sau ${stepIdx} bước. Cánh tay đã về vị trí nghỉ.`;
    }
  }

  // 6. Update Objects Roster Coordinates
  if (data.objects) {
    state.allObjects = data.objects;
    updateObjectRosterCoordinates(data.objects);
  }
}

function updateObjectRosterCoordinates(objects) {
  for (const [key, obj] of Object.entries(objects)) {
    const coordEl = document.getElementById(`coord-${key}`);
    if (coordEl && obj.pos) {
      coordEl.textContent = `[${obj.pos[0].toFixed(2)}, ${obj.pos[1].toFixed(2)}, ${obj.pos[2].toFixed(2)}]`;
    }
  }
}


// ============================================================================
// 3. CHAT CONSOLE LOGIC & NATURAL LANGUAGE COMMANDS
// ============================================================================
chatForm.addEventListener('submit', (e) => {
  e.preventDefault();
  const text = chatInput.value.trim();
  if (!text) return;
  sendChatMessage(text);
  chatInput.value = '';
});

async function sendChatMessage(userMessage) {
  // 1. Render User Message in Feed
  appendUserMessage(userMessage);

  // 2. Call /api/chat
  try {
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: userMessage })
    });

    if (!resp.ok) {
      throw new Error(`Server returned HTTP ${resp.status}`);
    }

    const data = await resp.json();
    appendRobotMessage(data);

    // Update active target if parsed
    if (data.target_key) {
      setActiveTargetUI(data.target_key);
    }
  } catch (err) {
    console.error('[Chat] Error sending message:', err);
    appendSystemErrorMessage(`Lỗi giao tiếp: ${err.message}`);
  }
}

function appendUserMessage(text) {
  const timeStr = getFormattedTime();
  const bubble = document.createElement('div');
  bubble.className = 'message-bubble msg-user';
  bubble.innerHTML = `
    <div class="msg-avatar">U</div>
    <div class="msg-content">
      <div class="msg-author">Bạn <span class="msg-timestamp">${timeStr}</span></div>
      <p>${escapeHtml(text)}</p>
    </div>
  `;
  chatFeed.appendChild(bubble);
  scrollChatToBottom();
}

function appendRobotMessage(data) {
  const timeStr = getFormattedTime();
  const bubble = document.createElement('div');
  bubble.className = 'message-bubble msg-robot';

  let reasoningHtml = '';
  if (Array.isArray(data.reasoning_steps) && data.reasoning_steps.length > 0) {
    const stepsList = data.reasoning_steps
      .map(step => `
        <div class="step-item">
          <span class="step-bullet">•</span>
          <span>${escapeHtml(step)}</span>
        </div>
      `).join('');

    reasoningHtml = `
      <div class="reasoning-box">
        <div class="reasoning-title">
          Suy luận Quỹ đạo &amp; Thao tác:
        </div>
        <div class="reasoning-steps">
          ${stepsList}
        </div>
      </div>
    `;
  }

  let modeBadge = '';
  if (data.is_offline_fallback || data.control_paradigm === 'offline_fallback') {
    modeBadge = `<span class="badge-tag" style="background: rgba(245, 158, 11, 0.2); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.4); margin-left: 8px; font-size: 10px; padding: 2px 6px; border-radius: 4px;">OFFLINE FALLBACK</span>`;
  } else if (data.control_paradigm === 'vlm_ik') {
    modeBadge = `<span class="badge-tag" style="background: rgba(6, 182, 212, 0.2); color: #06b6d4; border: 1px solid rgba(6, 182, 212, 0.4); margin-left: 8px; font-size: 10px; padding: 2px 6px; border-radius: 4px;">VLM + IK</span>`;
  } else if (data.vla_mode_tag) {
    modeBadge = `<span class="badge-tag" style="background: rgba(16, 185, 129, 0.2); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.4); margin-left: 8px; font-size: 10px; padding: 2px 6px; border-radius: 4px;">${escapeHtml(data.vla_mode_tag)}</span>`;
  }

  bubble.innerHTML = `
    <div class="msg-avatar">FP</div>
    <div class="msg-content">
      <div class="msg-author">${escapeHtml(state.vlaModel)} Agent ${modeBadge} <span class="msg-timestamp">${timeStr}</span></div>
      <p>${formatMarkdown(data.reply || '')}</p>
      ${reasoningHtml}
    </div>
  `;
  chatFeed.appendChild(bubble);
  scrollChatToBottom();
}

function appendSystemErrorMessage(msg) {
  const bubble = document.createElement('div');
  bubble.className = 'message-bubble msg-robot';
  bubble.innerHTML = `
    <div class="msg-avatar" style="background: rgba(244, 63, 94, 0.15); color: var(--accent-rose); border-color: rgba(244, 63, 94, 0.3); font-weight: 700;">
      !
    </div>
    <div class="msg-content" style="border-color: rgba(244, 63, 94, 0.3);">
      <div class="msg-author" style="color: var(--accent-rose);">Hệ thống</div>
      <p style="color: var(--accent-rose);">${escapeHtml(msg)}</p>
    </div>
  `;
  chatFeed.appendChild(bubble);
  scrollChatToBottom();
}

function scrollChatToBottom() {
  chatFeed.scrollTop = chatFeed.scrollHeight;
}

// Quick Prompt Chips Click
chipsContainer.addEventListener('click', (e) => {
  const chip = e.target.closest('.prompt-chip');
  if (!chip) return;
  const prompt = chip.getAttribute('data-prompt');
  if (prompt) {
    sendChatMessage(prompt);
  }
});

// Quick Action Buttons
btnQuickRandomize.addEventListener('click', async () => {
  try {
    const res = await fetch('/api/reposition_all', { method: 'POST' });
    appendChatMessage('system', 'Đã xáo trộn ngẫu nhiên vị trí 5 vật thể trên mặt bàn.');
    fetchSnapshot();
  } catch (err) {
    console.error(err);
  }
});

if (btnQuickReset) {
  btnQuickReset.addEventListener('click', async () => {
    try {
      await fetch('/api/reset', { method: 'POST' });
      sendChatMessage('Reset physics về trạng thái initial');
    } catch (err) {
      console.error(err);
    }
  });
}

btnQuickOpenGripper.addEventListener('click', () => {
  sendChatMessage('Mở kẹp');
});

btnQuickCloseGripper.addEventListener('click', () => {
  sendChatMessage('Đóng kẹp');
});


// ============================================================================
// 4. DUAL-CAMERA VIEWPORT CONTROLS
// ============================================================================
btnViewDual.addEventListener('click', () => setViewLayout('dual'));
btnViewTop.addEventListener('click', () => setViewLayout('top'));
btnViewWrist.addEventListener('click', () => setViewLayout('wrist'));

function setViewLayout(mode) {
  state.viewLayout = mode;
  viewportContainer.setAttribute('data-layout', mode);

  btnViewDual.classList.toggle('active', mode === 'dual');
  btnViewTop.classList.toggle('active', mode === 'top');
  btnViewWrist.classList.toggle('active', mode === 'wrist');
}

btnDismissBanner.addEventListener('click', () => {
  taskResultOverlay.classList.add('hidden');
});


// ============================================================================
// 5. OBJECTS ROSTER & TARGET SELECTION
// ============================================================================
rosterChipsContainer.addEventListener('click', async (e) => {
  const chip = e.target.closest('.roster-chip');
  if (!chip || chip.classList.contains('roster-basket')) return;

  const targetKey = chip.getAttribute('data-target');
  if (!targetKey) return;

  setActiveTargetUI(targetKey);

  // Inform backend
  try {
    await fetch('/api/set_target', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ target: targetKey })
    });

    const targetNames = {
      can: 'lon súp Campbell',
      mustard: 'chai mù tạt French',
      blue_box: 'hộp xanh dương',
      red_cylinder: 'trụ đỏ',
      green_cube: 'khối xanh lá'
    };
    sendChatMessage(`Gắp ${targetNames[targetKey] || targetKey} vào rỏ`);
  } catch (err) {
    console.error('Failed to set target:', err);
  }
});

function setActiveTargetUI(targetKey) {
  state.activeTarget = targetKey;
  const chips = rosterChipsContainer.querySelectorAll('.roster-chip');
  chips.forEach(c => {
    if (c.getAttribute('data-target') === targetKey) {
      c.classList.add('active');
    } else {
      c.classList.remove('active');
    }
  });
}


// ============================================================================
// 6. VLA MODEL SELECTION & READINESS GHOSTING
// ============================================================================
async function initVlaModels() {
  try {
    const resp = await fetch('/api/models_status');
    const data = await resp.json();
    if (data.status === 'ok' && Array.isArray(data.models)) {
      renderVlaModelOptions(data.models, data.active_model);
    }
  } catch (err) {
    console.warn('[VLA] Could not fetch models status:', err);
  }
}

function renderVlaModelOptions(models, activeKey) {
  vlaModelSelect.innerHTML = '';
  models.forEach(m => {
    const opt = document.createElement('option');
    opt.value = m.id;
    const labelBadge = m.badge && m.badge.includes('Ready') ? ` [${m.badge}]` : '';
    opt.textContent = m.ready ? `${m.name}${labelBadge}` : `${m.name} (API required)`;
    if (!m.ready) {
      opt.disabled = true;
      opt.className = 'opt-ghosted';
    }
    if (m.id === activeKey) {
      opt.selected = true;
      state.vlaModel = m.name;
      chatAgentNameEl.textContent = m.name;
    }
    vlaModelSelect.appendChild(opt);
  });
}

vlaModelSelect.addEventListener('change', async (e) => {
  const modelKey = e.target.value;
  try {
    const resp = await fetch('/api/set_model', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: modelKey, api_url: state.remoteApiUrl })
    });

    const data = await resp.json();
    if (data.status === 'ok') {
      state.vlaModel = data.model;
      chatAgentNameEl.textContent = data.model;
      appendRobotMessage({
        reply: `Đã kích hoạt mô hình chính sách **${data.model}**. Hệ thống MuJoCo đã reset về vị trí xuất phát.`,
        reasoning_steps: [`Đang sử dụng VLA Model: ${data.model} (${data.connection_status}).`]
      });
    } else {
      appendSystemErrorMessage(`Không thể chuyển model: ${data.message}`);
    }
  } catch (err) {
    console.error('Error switching VLA model:', err);
  }
});

// API Config Modal
btnConfigApi.addEventListener('click', () => {
  apiModal.classList.remove('hidden');
  inputApiUrl.value = state.remoteApiUrl;
});

btnCloseModal.addEventListener('click', () => apiModal.classList.add('hidden'));
btnCancelModal.addEventListener('click', () => apiModal.classList.add('hidden'));

btnSaveApi.addEventListener('click', async () => {
  const apiUrl = inputApiUrl.value.trim();
  const geminiKey = inputGeminiKey ? inputGeminiKey.value.trim() : '';
  state.remoteApiUrl = apiUrl;

  try {
    const resp = await fetch('/api/config_keys', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ gemini_key: geminiKey, api_url: apiUrl })
    });
    const data = await resp.json();
    if (data.status === 'ok') {
      if (Array.isArray(data.models_status)) {
        renderVlaModelOptions(data.models_status, state.vlaModelKey || 'octo');
      }
      appendRobotMessage({
        reply: 'Đã lưu và cập nhật cấu hình API vào file `.env` thành công.',
        reasoning_steps: [
          geminiKey ? 'Đã kích hoạt khóa Gemini API Key.' : 'Chưa cấu hình Gemini API Key.',
          apiUrl ? `Đã cấu hình Remote GPU Endpoint: ${apiUrl}` : 'Remote GPU Endpoint: Mặc định cục bộ (RTX 4060 GPU / CPU).'
        ]
      });
    }
  } catch (err) {
    console.error('Failed to save API config:', err);
  }

  apiModal.classList.add('hidden');
  await initVlaModels();
});


// ============================================================================
// 7. CONTROL PARADIGM SWITCHER (VLA, VLM + IK, Offline Fallback)
// ============================================================================
if (btnParadigmVla) btnParadigmVla.addEventListener('click', () => setControlParadigm('vla'));
if (btnParadigmVlmIk) btnParadigmVlmIk.addEventListener('click', () => setControlParadigm('vlm_ik'));
if (btnParadigmFallback) btnParadigmFallback.addEventListener('click', () => setControlParadigm('offline_fallback'));

async function setControlParadigm(paradigm) {
  try {
    const resp = await fetch('/api/paradigm', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ paradigm: paradigm })
    });
    const data = await resp.json();
    if (data.status === 'ok') {
      state.controlParadigm = paradigm;
      updateParadigmUI(paradigm);

      const paradigmNames = {
        vla: 'VLA (End-to-End Direct Policy)',
        vlm_ik: 'VLM + IK (Spatial Grounding + Kinematics)',
        offline_fallback: 'Offline Fallback (Analytical Servoing)'
      };

      appendRobotMessage({
        reply: `Đã kích hoạt phương thức điều khiển: **${paradigmNames[paradigm] || paradigm}**.`,
        reasoning_steps: [
          paradigm === 'vla'
            ? 'Trực tiếp dự đoán hành động từ Camera RGB + Text bằng mô hình VLA (SmolVLA / Octo).'
            : (paradigm === 'vlm_ik'
              ? 'VLM nhận thức vị trí vật thể + giải Inverse Kinematics mượt mà.'
              : 'Bộ nội suy chuyển động giải tích không cần mô hình học máy.')
        ],
        control_paradigm: paradigm
      });
    }
  } catch (err) {
    console.error('Failed to switch paradigm:', err);
  }
}

function updateParadigmUI(paradigm) {
  if (btnParadigmVla) btnParadigmVla.classList.toggle('active', paradigm === 'vla');
  if (btnParadigmVlmIk) btnParadigmVlmIk.classList.toggle('active', paradigm === 'vlm_ik');
  if (btnParadigmFallback) btnParadigmFallback.classList.toggle('active', paradigm === 'offline_fallback');
}


// ============================================================================
// 8. ABLATION MODE SWITCHING (+Memory vs Pure Reactive)
// ============================================================================
btnAblationMemory.addEventListener('click', () => setAblationMode('memory'));
btnAblationRaw.addEventListener('click', () => setAblationMode('raw'));

async function setAblationMode(mode) {
  try {
    const resp = await fetch('/api/ablation_mode', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode: mode })
    });
    const data = await resp.json();
    if (data.status === 'ok') {
      state.ablationMode = mode;
      btnAblationMemory.classList.toggle('active', mode === 'memory');
      btnAblationRaw.classList.toggle('active', mode === 'raw');

      const modeName = mode === 'memory' ? '+Memory' : 'Pure Reactive';
      appendRobotMessage({
        reply: `Đã chuyển sang chế độ **${modeName}**.`,
        reasoning_steps: [
          mode === 'memory'
            ? 'Kích hoạt bộ neo TAPIR và Chronicler Dynamic Prompting.'
            : 'Vô hiệu hóa lớp nhớ: Chạy thuần RGB.'
        ]
      });
    }
  } catch (err) {
    console.error('Failed to set ablation mode:', err);
  }
}


// ============================================================================
// 8. STRING & TIME UTILITIES
// ============================================================================
function getFormattedTime() {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, '0');
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function formatMarkdown(str) {
  if (!str) return '';
  let out = escapeHtml(str);
  out = out.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  out = out.replace(/\*([^*]+)\*/g, '<em>$1</em>');
  return out;
}


// ============================================================================
// 8.5 LIBERO MINI-SUITE BENCHMARK MANAGEMENT
// ============================================================================
async function initLiberoSuite() {
  try {
    const res = await fetch('/api/libero/tasks');
    const data = await res.json();
    if (data.status === 'ok') {
      state.liberoTiers = data.tiers || {};
      state.liberoCurrentTier = data.current_tier || 1;
      state.liberoCurrentTaskId = data.current_task_id || 0;
      state.envMode = data.env_mode || 'libero';

      updateEnvModeUI(state.envMode);

      if (liberoTierSelect) {
        liberoTierSelect.value = String(state.liberoCurrentTier);
      }
      renderLiberoTasksForTier(state.liberoCurrentTier, state.liberoCurrentTaskId);
    }
  } catch (err) {
    console.error('Failed to init LIBERO suite:', err);
  }
}

function renderLiberoTasksForTier(tier, selectedTaskId = null) {
  const tierInfo = state.liberoTiers[tier];
  if (!tierInfo || !liberoTaskSelect) return;

  if (liberoSuiteBadge) {
    liberoSuiteBadge.textContent = tierInfo.badge || tierInfo.suite;
  }

  liberoTaskSelect.innerHTML = '';
  const tasks = tierInfo.tasks || [];
  tasks.forEach((task, idx) => {
    const opt = document.createElement('option');
    opt.value = task.id;
    opt.textContent = `${task.id}: ${task.name}`;
    if (selectedTaskId !== null && task.id === selectedTaskId) {
      opt.selected = true;
    } else if (selectedTaskId === null && idx === 0) {
      opt.selected = true;
    }
    liberoTaskSelect.appendChild(opt);
  });

  const activeTask = tasks.find(t => t.id === Number(liberoTaskSelect.value)) || tasks[0];
  if (activeTask) {
    state.liberoCurrentTaskId = activeTask.id;
    state.liberoInstruction = activeTask.instruction;
    if (liberoDefaultPrompt) {
      liberoDefaultPrompt.textContent = activeTask.instruction;
    }
  }
}

if (liberoTierSelect) {
  liberoTierSelect.addEventListener('change', async (e) => {
    const tier = Number(e.target.value);
    state.liberoCurrentTier = tier;
    renderLiberoTasksForTier(tier);
    const firstTask = state.liberoTiers[tier]?.tasks[0];
    if (firstTask) {
      await setLiberoTask(tier, firstTask.id);
    }
  });
}

if (liberoTaskSelect) {
  liberoTaskSelect.addEventListener('change', async (e) => {
    const taskId = Number(e.target.value);
    state.liberoCurrentTaskId = taskId;
    const tierInfo = state.liberoTiers[state.liberoCurrentTier];
    const taskObj = tierInfo?.tasks.find(t => t.id === taskId);
    if (taskObj && liberoDefaultPrompt) {
      liberoDefaultPrompt.textContent = taskObj.instruction;
      state.liberoInstruction = taskObj.instruction;
    }
    await setLiberoTask(state.liberoCurrentTier, taskId);
  });
}

async function setLiberoTask(tier, taskId) {
  try {
    const res = await fetch('/api/libero/set_task', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tier, task_id: taskId })
    });
    const data = await res.json();
    if (data.status === 'ok') {
      appendRobotMessage({
        reply: `Đã nạp bối cảnh chuẩn **LIBERO Tier ${tier} (Task ${taskId})**: "${data.instruction}"`,
        reasoning_steps: [
          `Thiết lập môi trường MuJoCo BDDL cho Tier ${tier}`,
          `Nhiệm vụ: ${data.task_name}`,
          `Prompt tiêu chuẩn: "${data.instruction}"`,
          'Khôi phục tư thế ban đầu của Franka Panda và các vật thể.'
        ]
      });
      fetchSnapshot();
    }
  } catch (err) {
    console.error('Failed to set LIBERO task:', err);
  }
}

if (btnUseDefaultPrompt) {
  btnUseDefaultPrompt.addEventListener('click', () => {
    if (liberoDefaultPrompt && chatInput) {
      chatInput.value = liberoDefaultPrompt.textContent;
      chatInput.focus();
    }
  });
}

if (btnLiberoExecute) {
  btnLiberoExecute.addEventListener('click', async () => {
    const instruction = chatInput.value.trim() || liberoDefaultPrompt.textContent.trim();
    try {
      appendUserMessage(instruction || 'Thực thi tác vụ LIBERO');
      const res = await fetch('/api/libero/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ instruction })
      });
      const data = await res.json();
      if (data.status === 'ok') {
        appendRobotMessage({
          reply: `⚡ [MiniVLA VQ-Libero90] Bắt đầu thực thi: "${data.instruction}"...`,
          reasoning_steps: [
            'Nạp camera frames: agentview & robot0_eye_in_hand',
            'Sinh chuỗi hành động Action Chunks 8 bước (7-DoF) qua VQ-VAE tokenizer',
            'Thực thi điều khiển Franka Panda thời gian thực với tần số 20Hz'
          ]
        });
      }
    } catch (err) {
      console.error('Failed to execute LIBERO task:', err);
    }
  });
}

if (btnLiberoStop) {
  btnLiberoStop.addEventListener('click', async () => {
    try {
      await fetch('/api/libero/stop', { method: 'POST' });
      appendChatMessage('system', 'Đã dừng robot Franka Panda.');
    } catch (err) {
      console.error(err);
    }
  });
}

if (btnLiberoReset) {
  btnLiberoReset.addEventListener('click', async () => {
    try {
      await fetch('/api/libero/reset', { method: 'POST' });
      appendChatMessage('system', 'Đã reset tác vụ LIBERO về initial state.');
      fetchSnapshot();
    } catch (err) {
      console.error(err);
    }
  });
}

if (btnModeLibero) {
  btnModeLibero.addEventListener('click', () => setEnvMode('libero'));
}
if (btnModeTabletop) {
  btnModeTabletop.addEventListener('click', () => setEnvMode('tabletop'));
}

async function setEnvMode(mode) {
  try {
    const res = await fetch('/api/env_mode', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode })
    });
    const data = await res.json();
    if (data.status === 'ok') {
      state.envMode = data.env_mode;
      updateEnvModeUI(data.env_mode);
      appendChatMessage('system', `Đã chuyển sang chế độ: **${data.env_mode === 'libero' ? 'LIBERO Benchmark Mini-Suite' : 'Custom Tabletop'}**`);
      fetchSnapshot();
    }
  } catch (err) {
    console.error('Failed to switch env mode:', err);
  }
}

function updateEnvModeUI(mode) {
  const isLibero = mode === 'libero';
  if (btnModeLibero) btnModeLibero.classList.toggle('active', isLibero);
  if (btnModeTabletop) btnModeTabletop.classList.toggle('active', !isLibero);

  if (liberoSuitePanel) {
    liberoSuitePanel.style.display = isLibero ? 'block' : 'none';
  }
  if (chipsContainer) {
    chipsContainer.style.display = isLibero ? 'none' : 'flex';
  }
  const rosterBar = document.querySelector('.objects-roster-bar');
  if (rosterBar) {
    rosterBar.style.display = isLibero ? 'none' : 'flex';
  }

  if (camTopLabel) {
    camTopLabel.textContent = isLibero ? 'agentview (Camera Trước 45°)' : 'Top Camera';
  }
  if (camWristLabel) {
    camWristLabel.textContent = isLibero ? 'robot0_eye_in_hand (Camera Cổ Tay)' : 'Wrist Camera';
  }

  const titleEl = document.getElementById('console-header-title');
  if (titleEl) {
    titleEl.textContent = isLibero ? 'LIBERO Benchmark & Chat Console' : 'Chat Console';
  }
  if (chatAgentNameEl) {
    chatAgentNameEl.textContent = isLibero ? 'MiniVLA (VQ-Libero90)' : state.vlaModel;
  }
}


// ============================================================================
// 9. INITIALIZATION
// ============================================================================
window.addEventListener('DOMContentLoaded', () => {
  console.log('[Dashboard] Initializing Franka Panda MuJoCo VLA Workspace...');
  // Immediately request snapshot for instant frame display before SSE starts
  fetch('/api/snapshot')
    .then(r => r.json())
    .then(data => {
      if (data && data.frame_top_b64) handleStreamPayload(data);
    })
    .catch(() => {});
  connectSSEStream();
  initVlaModels();
  initLiberoSuite();
});
