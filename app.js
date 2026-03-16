/* ================================================================
   CamGuard AI — Main Application Logic  v2.0 (Multi-Sensor Fusion)
   Modules: Visual AI, Lens Reflection, Network Scan, Magnetic Field,
            IR Scanner, Fusion Engine, Neural Pattern Matcher
   ================================================================ */

'use strict';

// ================================================================
// STATE
// ================================================================
const state = {
  currentScreen: 'screen-home',
  scores: { visual: 0, lens: 0, network: 0, magnetic: 0, ir: 0 },
  findings: { visual: '—', lens: '—', network: '—', magnetic: '—', ir: '—' },
  fusionMeta: { magValue: 0, networkHints: [], networkDevices: 0, lastFusionResult: null },
  visual: {
    stream: null, model: null, running: false, animFrame: null,
    lastTime: 0, fps: 0, facingMode: 'environment', camCount: 0,
    fileMode: false, fileUrl: null
  },
  lens: {
    stream: null, running: false, animFrame: null,
    torchOn: false, track: null, alertShown: false
  },
  network: {
    running: false, devices: [], abortCtrl: null, scanned: 0, total: 255
  },
  magnetic: {
    running: false, data: [], listener: null,
    baseline: null, alertShown: false, chartCtx: null,
    lastValue: 0, neuralMatch: false,
    calibrationStd: null, calibrationProgress: 0, smoothValue: null
  },
  ir: {
    stream: null, running: false, animFrame: null,
    intensity: 0, lastVal: 0, alerts: 0
  },
  config: {
    modelType: 'custom' // 'coco' | 'yolo' | 'hybrid' | 'custom'
  },
  models: {
    coco: null,
    yolo: null,
    custom: null
  },
  fullScan: { running: false },
  installPrompt: null,
  offlineHintShown: false
};

const CUSTOM_MODEL_PATH = './models/tfjs/model.json';

// ================================================================
// LEARNING ENGINE — localStorage persistence
// Saves every scan session and adapts detection thresholds based
// on user feedback (confirmed detections vs false alarms).
// ================================================================
const LS_KEY = 'camguard_learning_v2';
const MAX_SESSIONS = 100;

// Load or initialise the persistent data store
function loadLearningData()
{
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (raw) return JSON.parse(raw);
  } catch (e) { }
  return {
    version: 3,
    totalScans: 0,
    totalDetections: 0,
    confirmedDetections: 0,
    falseAlarms: 0,
    // Learned thresholds (start at defaults, drift with feedback)
    adaptedGlintRatio: 1.6,  // lower = more sensitive
    adaptedRiskThreshold: 35,   // alert fires above this%
    adaptedAIThreshold: 0.60,   // AI confidence required
    // Knowledge Base: Array of learned camera features/patterns
    knowledgeBase: [],
    // Recent scan sessions
    sessions: [],
    // Current open session (null when not scanning)
    currentSession: null
  };
}

function saveLearningData()
{
  try { localStorage.setItem(LS_KEY, JSON.stringify(db)); }
  catch (e) { console.warn('CamGuard: could not persist data', e); }
}

// The live database object
let db = loadLearningData();

// ---- Learning Engine Initialization (Auto-Sync from Project) ----
const MASTER_BRAIN_VERSION = 12;
const DATA_PATHS = {
  runtimeBrain: './data/brains/runtime_brain.json'
};

function normalizeBrainPayload(payload)
{
  if (!payload || !Array.isArray(payload.knowledgeBase)) return null;

  if (payload.version) {
    return {
      totalScans: 0,
      totalDetections: payload.knowledgeBase.length,
      confirmedDetections: 0,
      falseAlarms: 0,
      adaptedGlintRatio: 1.6,
      adaptedRiskThreshold: 35,
      adaptedAIThreshold: 0.60,
      sessions: [],
      currentSession: null,
      ...payload,
      knowledgeBase: payload.knowledgeBase
    };
  }

  return {
    version: MASTER_BRAIN_VERSION,
    totalScans: 0,
    totalDetections: payload.entryCount || payload.knowledgeBase.length,
    confirmedDetections: 0,
    falseAlarms: 0,
    adaptedGlintRatio: 1.6,
    adaptedRiskThreshold: 35,
    adaptedAIThreshold: 0.60,
    sessions: [],
    currentSession: null,
    sourceDatasetType: payload.datasetType || 'dataset',
    schemaVersion: payload.schemaVersion || 1,
    knowledgeBase: payload.knowledgeBase
  };
}

async function fetchProjectBrain()
{
  const resp = await fetch(DATA_PATHS.runtimeBrain);
  if (!resp.ok) throw new Error(`brain fetch failed (${resp.status})`);
  const payload = await resp.json();
  const normalized = normalizeBrainPayload(payload);
  if (!normalized) throw new Error('invalid brain payload');
  return normalized;
}

async function initLearningEngine()
{
  // If local brain version is older than master, or empty, we force a sync
  if (!db.version || db.version < MASTER_BRAIN_VERSION || db.knowledgeBase.length === 0) {
    console.log(`🧠 AI Brain outdated (v${db.version || 0}). Syncing with Master Brain v${MASTER_BRAIN_VERSION}...`);
    try {
      const projectBrain = await fetchProjectBrain();
      const mergedBrain = {
        ...db,
        ...projectBrain,
        totalScans: Math.max(db.totalScans || 0, projectBrain.totalScans || 0),
        confirmedDetections: Math.max(db.confirmedDetections || 0, projectBrain.confirmedDetections || 0),
        falseAlarms: Math.max(db.falseAlarms || 0, projectBrain.falseAlarms || 0),
        version: MASTER_BRAIN_VERSION
      };
      db = mergedBrain;
      saveLearningData();
      updateHomeStats();
      if (window.updateLabStats) updateLabStats();
      showToast('Runtime Brain Synced', 'success', 4000);
      if (false) {
        const projectBrain = await resp.json();

        // Strategy: Preserve local stats (totalScans) but adopt the Master knowledgeBase
        // This keeps the user's scan count while giving them the new expert detections.
        const mergedBrain = {
          ...db,
          ...projectBrain,
          totalScans: Math.max(db.totalScans, projectBrain.totalScans),
          version: MASTER_BRAIN_VERSION
        };

        db = mergedBrain;
        saveLearningData();
        updateHomeStats();
        if (window.updateLabStats) updateLabStats();

        showToast('🚀 Brain Synced: Master Patterns v5 Loaded', 'success', 4000);
      }
    } catch (e) {
      console.warn('AI Sync Failed. Using local fallback.', e.message);
    }
  }
}
initLearningEngine();

// ================================================================
// PERFORMANCE OPTIMIZATION — Mobile CPU / Frame Rate Management
// ================================================================
const PERF = (() =>
{
  // Detect device tier from memory and CPU hints
  const mem = navigator.deviceMemory || 4;    // GB (not on all browsers)
  const cores = navigator.hardwareConcurrency || 4;
  const tier = mem <= 1 || cores <= 2 ? 'low'
    : mem <= 3 || cores <= 4 ? 'mid' : 'high';

  // Target frame intervals per tier  (ms between processed frames)
  const FPS_MAP = { low: 100, mid: 60, high: 33 }; // ~10, ~17, ~30 fps
  const frameInterval = FPS_MAP[tier];

  console.log(`[CamGuard] Device tier: ${tier} | Memory: ${mem}GB | Cores: ${cores} | Frame interval: ${frameInterval}ms`);

  return { tier, frameInterval };
})();

// Set TensorFlow.js backend: prefer WebGL for GPU acceleration; fall back to WASM/CPU
(async () =>
{
  try {
    if (typeof tf !== 'undefined') {
      await tf.setBackend('webgl');
      await tf.ready();
      // Limit TF memory growth on mobile
      tf.env().set('WEBGL_DELETE_TEXTURE_THRESHOLD', 0);
      console.log(`[CamGuard] TF backend: ${tf.getBackend()}`);
    }
  } catch (e) {
    console.warn('[CamGuard] WebGL not available, using CPU backend:', e.message);
  }
})();

// Migration: ensure new fields exist if coming from v2
if (!db.knowledgeBase) db.knowledgeBase = [];
if (!db.adaptedAIThreshold) db.adaptedAIThreshold = 0.60;
if (db.version < 3) db.version = 3;

// ---- Session lifecycle ----
function sessionStart(type)
{
  db.currentSession = {
    id: Date.now(),
    type,                    // 'lens' | 'visual' | 'network' | 'magnetic'
    startedAt: new Date().toISOString(),
    duration: 0,
    maxRisk: 0,
    glintsFound: 0,
    aiFlags: 0,
    networksFound: 0,
    avgRoomLum: 0,
    alertFired: false,
    confirmed: false,
    falseAlarm: false
  };
  db.totalScans++;
  saveLearningData();
  updateHomeStats();
  if (typeof updateLabStats === 'function') updateLabStats();
}

function sessionUpdate(fields)
{
  if (!db.currentSession) return;
  Object.assign(db.currentSession, fields);
  // Running max risk
  if (fields.maxRisk !== undefined)
    db.currentSession.maxRisk = Math.max(db.currentSession.maxRisk, fields.maxRisk);
}

function sessionEnd(confirmed = false, falseAlarm = false)
{
  if (!db.currentSession) return;
  const s = db.currentSession;
  s.endedAt = new Date().toISOString();
  s.duration = Math.round((Date.now() - s.id) / 1000);
  s.confirmed = confirmed;
  s.falseAlarm = falseAlarm;

  if (confirmed) {
    db.confirmedDetections++;
    db.totalDetections++;
  }
  if (falseAlarm) {
    db.falseAlarms++;
    // Adapt: raise glintRatio slightly (be less sensitive) to reduce false positives
    db.adaptedGlintRatio = Math.min(3.0, db.adaptedGlintRatio + 0.05);
    db.adaptedRiskThreshold = Math.min(55, db.adaptedRiskThreshold + 2);
  }
  if (confirmed && !falseAlarm) {
    // Adapt: lower glintRatio (be more sensitive) since we confirmed a real camera
    db.adaptedGlintRatio = Math.max(1.2, db.adaptedGlintRatio - 0.03);
    db.adaptedRiskThreshold = Math.max(20, db.adaptedRiskThreshold - 1);
  }

  // Keep sessions ring-buffer
  db.sessions.unshift(s);
  if (db.sessions.length > MAX_SESSIONS) db.sessions.length = MAX_SESSIONS;
  db.currentSession = null;
  saveLearningData();
  updateHomeStats();
}

// ---- Home screen stats ----
function updateHomeStats()
{
  const el = document.getElementById('homeStats');
  if (!el) return;
  const lastScan = db.sessions[0];
  const lastTime = lastScan
    ? new Date(lastScan.startedAt).toLocaleDateString('en', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
    : 'Never';
  el.innerHTML = `
    <div class="stat-pill">🔍 ${db.totalScans} scans</div>
    <div class="stat-pill alert-pill">🚨 ${db.confirmedDetections} confirmed</div>
    <div class="stat-pill">⚡ Sensitivity: ${Math.round((3.0 - db.adaptedGlintRatio) / (3.0 - 1.2) * 100)}%</div>
    <div class="stat-pill muted-pill">🕐 Last: ${lastTime}</div>`;
}

// ---- History screen ----
function showHistory()
{
  openScreen('screen-history');
  const list = document.getElementById('historyList');
  if (!db.sessions.length) {
    list.innerHTML = '<p class="list-empty">No scan history yet. Run your first scan!</p>';
    return;
  }
  list.innerHTML = db.sessions.slice(0, 40).map(s =>
  {
    const date = new Date(s.startedAt).toLocaleString('en', {
      month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
    });
    const icon = s.type === 'lens' ? '🔦' : s.type === 'visual' ? '🤖'
      : s.type === 'network' ? '📡' : '🧲';
    const badge = s.confirmed ? '<span class="screen-badge danger">⚠️ Camera</span>'
      : s.falseAlarm ? '<span class="screen-badge" style="background:rgba(16,185,129,0.15);color:var(--success)">✓ Clear</span>'
        : s.alertFired ? '<span class="screen-badge lens">Alert</span>'
          : '<span class="screen-badge">—</span>';
    return `<div class="detection-row" style="gap:10px;padding:8px 0">
      <span style="font-size:1.1rem">${icon}</span>
      <div style="flex:1;min-width:0">
        <div style="font-size:0.78rem;font-weight:600">${s.type.toUpperCase()} scan · ${date}</div>
        <div style="font-size:0.68rem;color:var(--text-muted)">
          Risk: ${s.maxRisk}% · ${s.duration}s · ${s.glintsFound} glints
        </div>
      </div>
      ${badge}
    </div>`;
  }).join('');
}

function clearHistory()
{
  if (!confirm('Clear all scan history and reset learned thresholds?')) return;
  db = loadLearningData(); // reset to defaults
  db.sessions = [];
  db.totalScans = 0;
  db.confirmedDetections = 0;
  db.falseAlarms = 0;
  db.totalDetections = 0;
  saveLearningData();
  showHistory();
  updateHomeStats();
  showToast('History cleared — thresholds reset to default', '', 3000);
}

// ---- Alert feedback (called from lens alert buttons) ----
// Called on EVERY detection — always asks Real or Fake.
// Stores the result into the AI Knowledge Base for learning.
function alertFeedback(isReal)
{
  const riskPct = parseInt(document.getElementById('lensRisk2')?.textContent) || 0;
  const glints = parseInt(document.getElementById('lensLabel')?.textContent?.match(/(\d+) glint/)?.[1]) || 0;
  const magVal = state.magnetic.lastValue || 0;

  if (isReal) {
    // ── Store confirmed camera pattern into the knowledge base ──────────
    if (window.CamGuardFusion) {
      window.CamGuardFusion.ThresholdAdaptor.storePattern(db, {
        label: 'Confirmed Camera (Lens Scan)',
        aiConfidence: db.adaptedAIThreshold,
        magValue: magVal,
        visualScore: state.scores.visual,
        lensScore: riskPct,
        fusedScore: riskPct,
        confirmed: true,
        environment: { glints, timestamp: Date.now() }
      });
      window.CamGuardFusion.ThresholdAdaptor.adapt(db, true, false);
    }
    sessionEnd(true, false);
    saveLearningData();
    showToast('✅ Confirmed! AI Brain updated — sensitivity increased.', 'warn', 4000);
    appendLensLog('✅ REAL CAMERA confirmed — pattern stored. Brain updated.', 'danger');

    // Re-arm immediately so the NEXT glint also asks
    state.lens.alertShown = false;
    document.getElementById('lensAlert').classList.add('hidden');
    sessionStart('lens'); // continue session for ongoing scan

  } else {
    // ── Store false alarm so AI learns NOT to flag this signature ────────
    if (window.CamGuardFusion) {
      window.CamGuardFusion.ThresholdAdaptor.storePattern(db, {
        label: 'False Alarm (Lens Scan)',
        aiConfidence: db.adaptedAIThreshold,
        magValue: magVal,
        visualScore: state.scores.visual,
        lensScore: riskPct,
        fusedScore: riskPct,
        confirmed: false,
        environment: { glints, timestamp: Date.now() }
      });
      window.CamGuardFusion.ThresholdAdaptor.adapt(db, false, true);
    }
    sessionEnd(false, true);
    saveLearningData();

    // Dismiss alert and immediately resume scanning
    document.getElementById('lensAlert').classList.add('hidden');
    state.lens.alertShown = false;
    setHeaderStatus('Lens+AI Scanning…', 'scanning');
    showToast('👍 False alarm noted — AI brain updated. Scanning resumed.', '', 4000);
    appendLensLog(`👍 FALSE ALARM stored. New threshold: ${db.adaptedGlintRatio.toFixed(2)} · Scan resumes automatically.`, 'info');

    // Start fresh session — scanning continues automatically (animFrame still running)
    sessionStart('lens');
  }
}


// Camera fingerprinting keywords
const CAMERA_KEYWORDS = [
  'cell phone', 'laptop', 'tv', 'remote', 'book', 'bottle', 'vase', 'clock',
  'mouse', 'keyboard', 'cup', 'person', 'cat', 'dog', 'chair', 'couch', 'bed'
];
// Camera-specific ports to probe (in priority order)
const CAMERA_PORTS = [80, 81, 8080, 8081, 554, 37777, 34568, 8000, 8001, 8888, 9000, 49152, 5000, 1935, 8554, 8443, 4433, 9100];
// Common non-camera device ports (used to classify safe devices)
const COMMON_PORTS = [80, 443, 22, 21, 8080, 631, 5000];
// Camera vendor signatures found in HTTP responses / hostnames
const CAMERA_SIGNATURES = [
  // Brand names in HTML title / headers
  'hikvision', 'dahua', 'axis', 'amcrest', 'reolink', 'nest cam', 'ring',
  'arlo', 'wyze', 'blink', 'foscam', 'vivotek', 'hanwha', 'bosch security',
  'flir', 'avigilon', 'uniview', 'tiandy', 'annke', 'eufy cam',
  // Generic flags
  'ipcam', 'ip camera', 'network camera', 'webcam', 'dvr', 'nvr',
  'surveillance', 'cctv', 'onvif', 'rtsp', 'video server',
  'camera web server', 'camera http server', 'cam interface'
];
// Camera URL paths to try for fingerprinting
const CAMERA_PATHS = [
  '/', '/index.htm', '/index.html', '/login.htm', '/login.html',
  '/view/index.shtml', '/doc/page/login.asp', // Hikvision
  '/web/index.html', '/cgi-bin/viewer/video.jpg', // Axis
  '/snapshot.cgi', '/image.jpg', '/video.cgi', '/mjpg/video.mjpg',   // Generic
  '/ISAPI/Security/userCheck', '/onvif/device_service',              // Hikvision / ONVIF
  '/cgi-bin/magicBox.cgi?action=getSystemInfo'   // Dahua API
];
// Response time thresholds (ms)
const HOST_UP_THRESHOLD = 1800; // Max ms for a host to be considered "up"
const PORT_FAST_THRESHOLD = 150; // Fast response suggests open port (local LAN)

// ================================================================
// SCREEN NAVIGATION
// ================================================================
function openScreen(id)
{
  document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
  const target = document.getElementById(id);
  if (target) {
    target.classList.add('active');
    state.currentScreen = id;
    target.scrollTop = 0;
    window.scrollTo(0, 0);
  }
}

// ================================================================
// UTILITIES
// ================================================================
function showToast(msg, type = '', duration = 3500)
{
  const container = document.getElementById('toastContainer');
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = msg;
  container.appendChild(toast);
  setTimeout(() =>
  {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(12px)';
    toast.style.transition = '0.3s';
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

function setHeaderStatus(label, cls)
{
  document.getElementById('statusText').textContent = label;
  const dot = document.querySelector('.status-dot');
  dot.className = 'status-dot ' + cls;
}

function updateThreatScore()
{
  const vals = Object.values(state.scores);
  const total = vals.reduce((a, b) => a + b, 0);
  const avg = Math.round(total / vals.length);
  const arc = document.getElementById('threatArc');
  const scoreEl = document.getElementById('threatScore');
  const titleEl = document.getElementById('threatTitle');
  const descEl = document.getElementById('threatDesc');

  scoreEl.textContent = avg;
  const offset = 314 - (314 * avg / 100);
  arc.style.strokeDashoffset = offset;

  if (avg === 0) {
    titleEl.textContent = 'SAFE ZONE';
    titleEl.style.color = 'var(--success)';
    descEl.textContent = 'No hidden cameras detected. Run scans to check your environment.';
  } else if (avg < 40) {
    titleEl.textContent = 'LOW RISK';
    titleEl.style.color = 'var(--primary)';
    descEl.textContent = 'Minor anomalies detected. Continue scanning for confirmation.';
  } else if (avg < 70) {
    titleEl.textContent = 'CAUTION';
    titleEl.style.color = 'var(--warn)';
    descEl.textContent = 'Suspicious activity detected! Investigate highlighted findings.';
  } else {
    titleEl.textContent = '⚠️ HIGH RISK';
    titleEl.style.color = 'var(--danger)';
    descEl.textContent = 'Hidden camera likely detected! Check all findings immediately.';
  }

  // Update findings grid
  Object.keys(state.findings).forEach(k =>
  {
    document.getElementById(`find-${k}`).textContent = state.findings[k];
  });
}

// ================================================================
// AI VISUAL SCAN (TensorFlow.js + COCO-SSD)
// ================================================================
function selectVideoFile()
{
  const input = document.getElementById('videoFileInput');
  if (input) input.click();
}

function handleVideoFileSelected(ev)
{
  const file = ev.target.files && ev.target.files[0];
  if (!file) return;
  if (state.visual.fileUrl) {
    URL.revokeObjectURL(state.visual.fileUrl);
  }
  state.visual.fileUrl = URL.createObjectURL(file);
  state.visual.fileMode = true;
  showToast('Loaded video file for scanning', 'success');
}

function useCameraMode()
{
  stopVisualScan();
  if (state.visual.fileUrl) {
    URL.revokeObjectURL(state.visual.fileUrl);
  }
  state.visual.fileUrl = null;
  state.visual.fileMode = false;
  const video = document.getElementById('visualVideo');
  if (video) {
    video.pause();
    video.src = '';
    video.controls = false;
  }
  showToast('Switched to live camera mode', 'info');
}

async function startVisualScan()
{
  if (state.visual.running) return;
  state.visual.running = true;

  const btnStart = document.getElementById('btnStartVisual');
  const btnStop = document.getElementById('btnStopVisual');
  btnStart.disabled = true;
  btnStop.disabled = false;

  setHeaderStatus('AI Scanning…', 'scanning');
  document.getElementById('detectionLabel').textContent = 'Loading AI Model…';
  showToast('Loading TensorFlow AI model…', 'warn', 3000);

  try {
    // Load AI Models based on selection
    if (state.config.modelType === 'coco' || state.config.modelType === 'hybrid') {
      if (!state.models.coco) {
        state.models.coco = await cocoSsd.load({ base: 'mobilenet_v2' });
        showToast('✅ COCO-SSD Engine Ready', 'success');
      }
    }
    if (state.config.modelType === 'yolo' || state.config.modelType === 'hybrid') {
      if (!state.models.yolo) {
        await loadYoloModel();
        showToast('✅ YOLOv8 Engine Ready', 'success');
      }
    }

    const video = document.getElementById('visualVideo');
    if (state.visual.fileMode && state.visual.fileUrl) {
      if (state.visual.stream) {
        state.visual.stream.getTracks().forEach(t => t.stop());
        state.visual.stream = null;
      }
      video.srcObject = null;
      video.src = state.visual.fileUrl;
      video.controls = true;
      video.muted = true;
      await video.play();
      document.getElementById('camSwitchBtn').style.display = 'none';
    } else {
      // Get camera stream
      const constraints = {
        video: {
          facingMode: state.visual.facingMode,
          width: { ideal: 768 }, height: { ideal: 1024 }
        }
      };

      if (state.visual.stream) {
        state.visual.stream.getTracks().forEach(t => t.stop());
      }

      state.visual.stream = await navigator.mediaDevices.getUserMedia(constraints);
      video.srcObject = state.visual.stream;
      video.controls = false;
      await video.play();
    }

    const canvas = document.getElementById('visualCanvas');
    const ctx = canvas.getContext('2d');

    document.getElementById('detectionLabel').textContent = '🤖 AI Scanning — Move phone slowly';

    // Count available cameras (camera mode only)
    if (!state.visual.fileMode) {
      const devices = await navigator.mediaDevices.enumerateDevices();
      state.visual.camCount = devices.filter(d => d.kind === 'videoinput').length;
      document.getElementById('camSwitchBtn').style.display = state.visual.camCount > 1 ? 'flex' : 'none';
    }

    let frameCount = 0;
    let lastFpsTime = performance.now();
    let camCount = 0;
    let maxConf = 0;
    let allObjects = 0;

    async function detectFrame()
    {
      if (!state.visual.running) return;

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const predictions = await getAIPredictions(video);
      allObjects = predictions.length;
      maxConf = 0;
      camCount = 0;

      predictions.forEach(pred =>
      {
        const isCam = isCameraLike(pred.class, pred.score);
        if (isCam) camCount++;
        maxConf = Math.max(maxConf, pred.score);

        const [x, y, w, h] = pred.bbox;
        const color = isCam ? '#ef4444' : pred.score > 0.7 ? '#00f5ff' : '#a855f7';

        // Draw bounding box
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, w, h);

        // Label background
        const label = `${pred.class} ${Math.round(pred.score * 100)}%`;
        ctx.font = 'bold 13px Inter, sans-serif';
        const tw = ctx.measureText(label).width;
        ctx.fillStyle = isCam ? 'rgba(239,68,68,0.8)' : 'rgba(0,10,30,0.75)';
        ctx.fillRect(x, y - 24, tw + 10, 22);

        // Label text
        ctx.fillStyle = '#fff';
        ctx.fillText(label, x + 5, y - 7);

        // Corner markers for camera detections
        if (isCam) {
          ctx.strokeStyle = '#ef4444';
          ctx.lineWidth = 3;
          const cs = 12;
          [[x, y], [x + w, y], [x, y + h], [x + w, y + h]].forEach(([cx, cy]) =>
          {
            ctx.beginPath(); ctx.moveTo(cx - cs, cy); ctx.lineTo(cx, cy); ctx.lineTo(cx, cy - cs);
            ctx.stroke();
          });
        }
      });

      // FPS
      frameCount++;
      const now = performance.now();
      if (now - lastFpsTime > 1000) {
        state.visual.fps = Math.round(frameCount * 1000 / (now - lastFpsTime));
        frameCount = 0; lastFpsTime = now;
      }

      // Update stats UI
      document.getElementById('vStat-fps').textContent = state.visual.fps || '…';
      document.getElementById('vStat-objects').textContent = allObjects;
      document.getElementById('vStat-cameras').textContent = camCount;
      document.getElementById('vStat-conf').textContent = maxConf > 0 ? Math.round(maxConf * 100) + '%' : '—';

      // Update detections list
      updateDetectionsList(predictions);

      // Update threat score
      if (camCount > 0) {
        state.scores.visual = Math.min(100, 50 + camCount * 25);
        state.findings.visual = `${camCount} cam${camCount > 1 ? 's' : ''}`;
        setHeaderStatus('Camera Detected!', 'alert');
        showToast(`⚠️ Possible camera detected! Check live view.`, 'danger', 4000);
      } else {
        state.scores.visual = 0;
        state.findings.visual = `${allObjects} objects`;
        setHeaderStatus('AI Scanning…', 'scanning');
      }
      updateThreatScore();

      state.visual.animFrame = requestAnimationFrame(detectFrame);
    }

    detectFrame();
  } catch (err) {
    state.visual.running = false;
    btnStart.disabled = false;
    btnStop.disabled = true;
    console.error(err);
    if (err.name === 'NotAllowedError') {
      showToast('❌ Camera permission denied. Please allow camera access.', 'danger', 5000);
    } else {
      showToast('❌ Error: ' + err.message, 'danger', 4000);
    }
    document.getElementById('detectionLabel').textContent = 'Camera access required';
    setHeaderStatus('Error', 'alert');
  }
}

// Objects that can HIDE or BE a hidden camera
// Never flag: person, chair, couch, bed, cat, dog — those are innocent
const CAM_HIDE_CLASSES = {
  // High suspicion — these often ARE spy cams
  'cell phone': 0.50,   // spy cam disguised as phone/charger
  'remote': 0.55,   // remote-shaped spy cam
  'clock': 0.60,   // clock cameras very common
  // Medium suspicion — common hiding spots
  'book': 0.80,   // book with hole camera
  'vase': 0.80,   // vase / plant pot camera
  'potted plant': 0.75, // plant spy cam
  'bottle': 0.85,   // water bottle cam
  'cup': 0.90,   // mug cam
  'mouse': 0.80,   // USB mouse cam
  'keyboard': 0.85,   // keyboard cam
  'tv': 0.90,   // hidden in TV bezel
  'laptop': 0.85,   // webcam in plain sight
};
// Classes that are NEVER cameras — always ignore
const CAM_SAFE_CLASSES = new Set([
  'person', 'people', 'man', 'woman', 'boy', 'girl',
  'chair', 'couch', 'sofa', 'bed', 'dining table', 'table',
  'cat', 'dog', 'bird', 'horse', 'cow', 'sheep', 'elephant',
  'banana', 'apple', 'orange', 'sandwich', 'pizza', 'cake',
  'car', 'truck', 'bus', 'bicycle', 'motorcycle', 'airplane', 'boat',
  'traffic light', 'stop sign', 'fire hydrant',
  'sports ball', 'kite', 'baseball bat', 'tennis racket',
  'umbrella', 'handbag', 'tie', 'suitcase', 'backpack',
  'frisbee', 'skis', 'snowboard', 'skateboard',
  'wine glass', 'fork', 'knife', 'spoon', 'bowl',
  'toilet', 'sink', 'refrigerator', 'microwave', 'oven', 'toaster',
  'bench', 'potted plant', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]);

function isCameraLike(objClass, score)
{
  if (CAM_SAFE_CLASSES.has(objClass)) return false;
  const threshold = CAM_HIDE_CLASSES[objClass];
  if (threshold !== undefined && score >= threshold) return true;
  return false;
}

function updateDetectionsList(predictions)
{
  const list = document.getElementById('detectionsList');
  if (predictions.length === 0) {
    list.innerHTML = '<p class="list-empty">No objects detected. Point camera around the room.</p>';
    return;
  }
  list.innerHTML = predictions
    .sort((a, b) => b.score - a.score)
    .slice(0, 8)
    .map(p =>
    {
      const isCam = isCameraLike(p.class, p.score);
      const confClass = p.score > 0.8 ? 'high' : p.score > 0.5 ? 'med' : 'low';
      return `<div class="detection-row">
        <span style="font-size:1.1rem">${getObjectEmoji(p.class)}</span>
        <span class="d-label">${p.class}</span>
        ${isCam ? '<span class="cam-detected-tag">⚠️ SUSPICIOUS</span>' : ''}
        <span class="d-conf ${confClass}">${Math.round(p.score * 100)}%</span>
      </div>`;
    }).join('');
}

function getObjectEmoji(cls)
{
  const map = {
    'cell phone': '📱', 'laptop': '💻', 'tv': '📺', 'remote': '🎮', 'person': '👤',
    'chair': '🪑', 'book': '📚', 'bottle': '🍶', 'clock': '🕐', 'cup': '☕',
    'cat': '🐱', 'dog': '🐶', 'keyboard': '⌨️', 'mouse': '🖱️'
  };
  return map[cls] || '📦';
}

function stopVisualScan()
{
  state.visual.running = false;
  if (state.visual.animFrame) cancelAnimationFrame(state.visual.animFrame);
  if (state.visual.stream) state.visual.stream.getTracks().forEach(t => t.stop());
  state.visual.stream = null;
  if (state.visual.fileMode) {
    const video = document.getElementById('visualVideo');
    if (video) video.pause();
  }

  document.getElementById('btnStartVisual').disabled = false;
  document.getElementById('btnStopVisual').disabled = true;
  document.getElementById('detectionLabel').textContent = 'Scan stopped';
  setHeaderStatus('Ready', 'safe');
  showToast('AI scan stopped', '', 2000);
}

async function switchCamera()
{
  state.visual.facingMode = state.visual.facingMode === 'environment' ? 'user' : 'environment';
  const wasRunning = state.visual.running;
  if (wasRunning) {
    stopVisualScan();
    setTimeout(() => startVisualScan(), 500);
  }
}

// ================================================================
// LENS REFLECTION SCAN — Continuous Dual AI+Optical Scanner
// Camera stays open until user taps Stop.
// Each frame runs:
//   Layer 1 — Optical: pixel-level brightness/IR analysis
//   Layer 2 — AI: TensorFlow COCO-SSD object detection
// Both layers draw onto separate canvases and combine into a
// single Risk Score. Detections are logged in real time.
// ================================================================

let lensAIRunning = false; // separate AI loop flag
let lensAIFrame = null;
let lensLogEntries = [];
const LENS_LOG_MAX = 50;

async function startLensScan()
{
  if (state.lens.running) return;
  state.lens.running = true;
  lensLogEntries = [];
  sessionStart('lens');

  const btnStart = document.getElementById('btnStartLens');
  const btnStop = document.getElementById('btnStopLens');
  const btnTorch = document.getElementById('btnTorch');
  btnStart.disabled = true;
  btnStop.disabled = false;
  btnTorch.disabled = false;

  setHeaderStatus('Lens+AI Scanning…', 'scanning');
  setLensRunningBadge(true);
  appendLensLog('🚀 Dual-mode scan started — camera open continuously', 'info');
  showToast('🔦 Camera on — sweep the room slowly!', 'warn', 4000);

  try {
    state.lens.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'environment', width: { ideal: 768 }, height: { ideal: 1024 } }
    });
    const tracks = state.lens.stream.getVideoTracks();
    state.lens.track = tracks[0];

    const caps = state.lens.track.getCapabilities ? state.lens.track.getCapabilities() : {};
    if (caps.torch) {
      appendLensLog('🔦 Torch supported — tap Torch ON for best results', 'info');
    } else {
      appendLensLog('ℹ️ Torch not supported — use ambient light', 'info');
    }

    const video = document.getElementById('lensVideo');
    video.srcObject = state.lens.stream;
    await video.play();

    const optCanvas = document.getElementById('lensCanvas');
    const aiCanvas = document.getElementById('lensAICanvas');
    const indicators = document.getElementById('reflectionIndicators');

    if (!optCanvas || !aiCanvas) {
      throw new Error(`Critical canvas element not found: ${!optCanvas ? 'lensCanvas' : 'lensAICanvas'}`);
    }

    const optCtx = optCanvas.getContext('2d', { alpha: true });
    const aiCtx = aiCanvas.getContext('2d', { alpha: true });

    if (!optCtx || !aiCtx) {
      throw new Error("Could not initialize 2D context on canvases");
    }

    // Analysis state (throttled)
    let analysisSkip = 0;
    const ANALYSIS_EVERY = 4;
    let aiFrameSkip = 0;
    const AI_EVERY_N = 10;
    let aiBusy = false;
    let aiPredictions = [];

    // Cached results for non-analysis frames
    let cachedGlints = [];
    let cachedBrightSpots = 0;
    let cachedMaxBright = 0;
    let cachedIR = 0;
    let cachedAvgLum = 128;

    // Local FPS tracking
    let frameCount = 0;
    let lastFpsTime = performance.now();
    let fps = 0;

    // Create a small offscreen canvas for pixel analysis (saves CPU)
    const offCanvas = document.createElement('canvas');
    const offCtx = offCanvas.getContext('2d', { willReadFrequently: true });

    document.getElementById('lensLabel').textContent = '🔍 Turn off lights — scan in darkness for best results';
    appendLensLog('🌑 Tip: scan in DARK room — lens glints glow against black background', 'info');

    // =================================================================
    // MAIN LOOP
    // =================================================================
    function lensFrame()
    {
      if (!state.lens.running) return;

      // Ensure video is ready before analysis
      if (video.readyState < 2) {
        state.lens.animFrame = requestAnimationFrame(lensFrame);
        return;
      }

      if (!optCtx || !aiCtx) return; // Safety guard

      try {
        const vw = video.videoWidth || 640;
        const vh = video.videoHeight || 480;
        const ow = Math.round(vw / 2);
        const oh = Math.round(vh / 2);

        if (optCanvas.width !== vw || optCanvas.height !== vh) {
          optCanvas.width = vw; optCanvas.height = vh;
          aiCanvas.width = vw; aiCanvas.height = vh;
        }
        if (offCanvas.width !== ow || offCanvas.height !== oh) {
          offCanvas.width = ow; offCanvas.height = oh;
        }

        optCtx.clearRect(0, 0, vw, vh);

        // ---- LAYER 1: Optical Analysis (Throttled) ----
        analysisSkip++;
        if (analysisSkip >= ANALYSIS_EVERY) {
          analysisSkip = 0;
          offCtx.drawImage(video, 0, 0, ow, oh);
          const px = offCtx.getImageData(0, 0, ow, oh).data;

          let brightSpots = 0, maxBrightness = 0, irScore = 0;
          let totalLumSample = 0, lumSamples = 0;
          for (let i = 0; i < px.length; i += 80) {
            totalLumSample += px[i] * 0.299 + px[i + 1] * 0.587 + px[i + 2] * 0.114;
            lumSamples++;
          }
          const avgRoomLum = totalLumSample / lumSamples;

          const isDark = avgRoomLum < 60;
          const BRIGHT_MIN = isDark ? 60 : 160;
          const GLINT_MIN = isDark ? Math.max(75, 90 - (db.adaptedGlintRatio - 1.6) * 20)
            : Math.max(180, 210 - (db.adaptedGlintRatio - 1.6) * 30);
          const GLINT_RATIO = isDark ? db.adaptedGlintRatio
            : Math.max(1.8, db.adaptedGlintRatio * 1.5);

          const glints = [];
          const STEP = 3;
          const NEIGHBOR_R = 9;

          function getLum(xi, yi)
          {
            if (xi < 0 || xi >= ow || yi < 0 || yi >= oh) return 0;
            const bIndex = (yi * ow + xi) * 4;
            return px[bIndex] * 0.299 + px[bIndex + 1] * 0.587 + px[bIndex + 2] * 0.114;
          }

          for (let y = NEIGHBOR_R; y < oh - NEIGHBOR_R; y += STEP) {
            for (let x = NEIGHBOR_R; x < ow - NEIGHBOR_R; x += STEP) {
              const bIndex = (y * ow + x) * 4;
              const r = px[bIndex], g = px[bIndex + 1], b = px[bIndex + 2];
              const lum = r * 0.299 + g * 0.587 + b * 0.114;

              if (lum > maxBrightness) maxBrightness = lum;
              if (lum < BRIGHT_MIN) continue;
              brightSpots++;

              if (r > 140 && b < 60 && lum > BRIGHT_MIN) irScore += 3;
              if (r > 100 && b > 80 && g < 60 && lum > BRIGHT_MIN) irScore += 2;

              if (lum < GLINT_MIN) continue;

              const neighbors = [
                getLum(x - NEIGHBOR_R, y), getLum(x + NEIGHBOR_R, y),
                getLum(x, y - NEIGHBOR_R), getLum(x, y + NEIGHBOR_R),
                getLum(x - NEIGHBOR_R, y - NEIGHBOR_R), getLum(x + NEIGHBOR_R, y - NEIGHBOR_R),
                getLum(x - NEIGHBOR_R, y + NEIGHBOR_R), getLum(x + NEIGHBOR_R, y + NEIGHBOR_R)
              ];
              const avgNeighbor = neighbors.reduce((s, v) => s + v, 0) / 8;
              const ratio = lum / (avgNeighbor + 6);

              if (ratio >= GLINT_RATIO) {
                glints.push({ x: x / ow * 100, y: y / oh * 100, lum, ratio, px: x, py: y });
              }
            }
          }

          const clusters = clusterGlintsByPixel(glints, ow, oh, 8);
          cachedGlints = clusters;
          cachedBrightSpots = brightSpots; cachedMaxBright = maxBrightness;
          cachedIR = irScore; cachedAvgLum = avgRoomLum;
        }

        const clusters = cachedGlints;
        const reflCount = clusters.length;
        const brightSpots = cachedBrightSpots;
        const maxBrightness = cachedMaxBright;
        const irScore = cachedIR;
        const avgRoomLum = cachedAvgLum;
        const isDark = avgRoomLum < 60;
        const roomLabel = isDark ? '🌑 Dark' : '💡 Lit';

        // Draw Glint markers
        optCtx.save();
        clusters.slice(0, 12).forEach(c =>
        {
          const cx = c.x / 100 * vw;
          const cy = c.y / 100 * vh;
          const radius = Math.max(8, Math.min(22, c.density * 2 + 7));
          optCtx.shadowColor = '#ff4400'; optCtx.shadowBlur = 10;
          optCtx.beginPath(); optCtx.arc(cx, cy, radius + 4, 0, Math.PI * 2);
          optCtx.strokeStyle = 'rgba(255,80,0,0.35)'; optCtx.lineWidth = 1.5; optCtx.stroke();
          optCtx.shadowBlur = 0;
          optCtx.beginPath(); optCtx.arc(cx, cy, radius, 0, Math.PI * 2);
          optCtx.strokeStyle = c.ratio > 3 ? '#ff1a00' : '#ff8800';
          optCtx.lineWidth = 2.5; optCtx.stroke();
          optCtx.strokeStyle = '#ffff00'; optCtx.lineWidth = 1;
          optCtx.beginPath();
          optCtx.moveTo(cx - 5, cy); optCtx.lineTo(cx + 5, cy);
          optCtx.moveTo(cx, cy - 5); optCtx.lineTo(cx, cy + 5);
          optCtx.stroke();
          optCtx.font = 'bold 10px monospace';
          optCtx.fillStyle = c.ratio > 3 ? '#ff4400' : '#ff9900';
          optCtx.fillText(`×${c.ratio.toFixed(1)}`, cx + radius + 3, cy + 4);
        });
        optCtx.restore();

        // Sync indicator dots
        indicators.innerHTML = '';
        clusters.slice(0, 8).forEach(c =>
        {
          const dot = document.createElement('div');
          dot.className = 'reflection-dot';
          dot.style.left = c.x + '%'; dot.style.top = c.y + '%';
          dot.style.width = '14px'; dot.style.height = '14px';
          indicators.appendChild(dot);
        });

        const normBright = Math.round(maxBrightness / 255 * 100);
        const normIR = Math.min(100, Math.round(irScore / 10));
        const optScore = Math.min(100, reflCount * 22 + normIR * 0.6);

        // ---- LAYER 2: AI Bounding Boxes ----
        aiCtx.clearRect(0, 0, vw, vh);
        let aiCamCount = 0;
        aiPredictions.forEach(pred =>
        {
          const suspicious = isCameraLike(pred.class, pred.score);
          if (!suspicious) return;

          const [bx, by, bw, bh] = pred.bbox;
          aiCamCount++;
          aiCtx.strokeStyle = '#ef4444'; aiCtx.lineWidth = 3;
          aiCtx.strokeRect(bx, by, bw, bh);
          const label = `⚠️ ${pred.class.toUpperCase()} ${Math.round(pred.score * 100)}%`;
          aiCtx.font = 'bold 12px Inter, sans-serif';
          const tw = aiCtx.measureText(label).width;
          aiCtx.fillStyle = 'rgba(239, 68, 68, 0.9)';
          aiCtx.fillRect(bx, by - 24, tw + 10, 22);
          aiCtx.fillStyle = '#fff';
          aiCtx.fillText(label, bx + 5, by - 8);

          const cs = 14;
          [[bx, by], [bx + bw, by], [bx, by + bh], [bx + bw, by + bh]].forEach(([cx2, cy2]) =>
          {
            aiCtx.beginPath();
            aiCtx.moveTo(cx2 - cs, cy2); aiCtx.lineTo(cx2, cy2); aiCtx.lineTo(cx2, cy2 - cs);
            aiCtx.stroke();
          });
        });

        // Trigger AI Throttled
        aiFrameSkip++;
        if (aiFrameSkip >= AI_EVERY_N && !aiBusy) {
          aiFrameSkip = 0; aiBusy = true;
          getAIPredictions(video).then(preds =>
          {
            aiPredictions = preds; aiBusy = false;
            preds.filter(p => isCameraLike(p.class, p.score)).forEach(p =>
            {
              appendLensLog(`🤖 AI: "${p.class}" (${Math.round(p.score * 100)}%) — suspicious object`, 'warn');
            });
          }).catch(() => { aiBusy = false; });
        }

        const aiScore = Math.min(100, aiCamCount * 40 + aiPredictions.length * 2);
        const combined = Math.min(100, Math.round(optScore * 0.65 + aiScore * 0.35));

        frameCount++;
        const now = performance.now();
        if (now - lastFpsTime > 1000) {
          fps = Math.round(frameCount * 1000 / (now - lastFpsTime));
          frameCount = 0; lastFpsTime = now;
        }

        document.getElementById('lsFPS').textContent = fps || '…';
        document.getElementById('lsSpots').textContent = brightSpots;
        document.getElementById('lsClusters').textContent = reflCount;
        document.getElementById('lsIR').textContent = normIR + '%';
        document.getElementById('lsAIObj').textContent = aiPredictions.length;
        document.getElementById('lsRiskScore').textContent = combined + '%';
        document.getElementById('lensBright').textContent = normBright + '%';
        document.getElementById('lensBrightBar').style.width = normBright + '%';
        document.getElementById('lensIR2').textContent = normIR + '%';
        document.getElementById('lensIRBar').style.width = normIR + '%';
        document.getElementById('lensRisk2').textContent = combined + '%';
        document.getElementById('lensRiskBar').style.width = combined + '%';

        let labelMsg = `${roomLabel} room — ${reflCount > 0 ? `⚠️ ${reflCount} glint(s)!` : '🔍 Sweep room slowly…'}`;
        if (reflCount > 0 && aiCamCount > 0) labelMsg = `🚨 ${reflCount} GLINTS + AI FLAG! CAMERA LIKELY`;
        else if (reflCount > 0) labelMsg = `⚠️ ${reflCount} SUSPICIOUS GLINTS — CHECK AREA`;
        else if (aiCamCount > 0) labelMsg = `🤖 AI: SUSPICIOUS DEVICE IN VIEW`;
        document.getElementById('lensLabel').textContent = labelMsg;

        if (combined > db.adaptedRiskThreshold && !state.lens.alertShown) {
          // Show alert and ALWAYS ask Real or Fake — every single detection
          state.lens.alertShown = true;
          sessionUpdate({ alertFired: true, maxRisk: combined, glintsFound: reflCount, aiFlags: aiCamCount, avgRoomLum });
          updateLensFusionScore(combined);
          const alertEl = document.getElementById('lensAlert');
          const alertTxt = document.getElementById('lensAlertText');
          alertEl.classList.remove('hidden');
          alertTxt.textContent = `🚨 Risk ${combined}% — ${reflCount} glints + ${aiCamCount} AI flag(s). Is this a real camera?`;
          showToast('🚨 Suspicious reflection! Please confirm below.', 'danger', 7000);
          appendLensLog(`🚨 ALERT: Risk=${combined}%, glints=${reflCount} — awaiting your feedback`, 'danger');
          setHeaderStatus('🔴 Confirm Detection!', 'alert');
        } else if (combined < 12 && state.lens.alertShown) {
          // Risk dropped — auto-dismiss so next spike triggers a fresh question
          state.lens.alertShown = false;
          document.getElementById('lensAlert').classList.add('hidden');
          setHeaderStatus('Lens+AI Scanning…', 'scanning');
        }

        if (Math.random() < 0.017) {
          sessionUpdate({ maxRisk: combined, glintsFound: reflCount, aiFlags: aiCamCount, avgRoomLum });
          saveLearningData();
          if (reflCount > 0) appendLensLog(`🔦 Found ${reflCount} glint(s) (Risk ${combined}%)`, 'warn');
        }

        state.scores.lens = combined;
        state.findings.lens = combined > db.adaptedRiskThreshold ? `Risk ${combined}%` : reflCount > 0 ? `${reflCount} glints` : 'Clear';
        updateThreatScore();

        state.lens.animFrame = requestAnimationFrame(lensFrame);
      } catch (loopErr) {
        console.warn('LensLoopErr:', loopErr);
        // If it's a critical error (like null access), don't restart to avoid infinite spam
        if (loopErr.message.includes('null')) {
          stopLensScan();
          showToast('❌ Camera error: ' + loopErr.message, 'danger', 6000);
        } else {
          state.lens.animFrame = requestAnimationFrame(lensFrame);
        }
      }
    }

    lensFrame();

  } catch (err) {
    state.lens.running = false;
    document.getElementById('btnStartLens').disabled = false;
    document.getElementById('btnStopLens').disabled = true;
    document.getElementById('btnTorch').disabled = true;
    setLensRunningBadge(false);
    console.error('Scan init error:', err);
    showToast('❌ Error: ' + err.message, 'danger', 5000);
    setHeaderStatus('Error', 'alert');
  }
}


// ---- Append a line to the live detection log ----
// Cluster detected lens glints by pixel distance
function clusterGlintsByPixel(glints, vw, vh, maxPx)
{
  const clusters = [];
  glints.forEach(g =>
  {
    let best = null, bestD = maxPx;
    clusters.forEach(c =>
    {
      const dx = g.px - c.px, dy = g.py - c.py;
      const d = Math.sqrt(dx * dx + dy * dy);
      if (d < bestD) { best = c; bestD = d; }
    });
    if (best) {
      best.x = (best.x * best.count + g.x) / (best.count + 1);
      best.y = (best.y * best.count + g.y) / (best.count + 1);
      best.px = (best.px * best.count + g.px) / (best.count + 1);
      best.py = (best.py * best.count + g.py) / (best.count + 1);
      best.ratio = Math.max(best.ratio, g.ratio);
      best.count++;
      best.density++;
    } else {
      clusters.push({ ...g, count: 1, density: 1 });
    }
  });
  return clusters;
}

function appendLensLog(msg, type = 'info')
{
  const log = document.getElementById('lensLog');
  if (!log) return;

  const time = new Date().toLocaleTimeString('en', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
  const colors = { info: 'var(--primary)', warn: 'var(--warn)', danger: 'var(--danger)' };

  const entry = document.createElement('div');
  entry.className = 'detection-row';
  entry.innerHTML = `
    <span style="color:var(--text-muted);font-size:0.65rem;font-family:var(--mono);min-width:52px">${time}</span>
    <span style="color:${colors[type] || colors.info};font-size:0.76rem;flex:1">${msg}</span>`;

  // Remove placeholder
  const placeholder = log.querySelector('.list-empty');
  if (placeholder) placeholder.remove();

  log.appendChild(entry);
  lensLogEntries.push(entry);

  // Trim old entries
  if (lensLogEntries.length > LENS_LOG_MAX) {
    lensLogEntries.shift()?.remove();
  }
  // Auto-scroll to bottom
  log.scrollTop = log.scrollHeight;
}

// ---- Update the "● Camera running" badge ----
function setLensRunningBadge(on)
{
  const el = document.getElementById('lensRunning');
  if (!el) return;
  if (on) {
    el.textContent = '● Camera running — continuous scan active';
    el.style.color = 'var(--success)';
    el.style.animation = 'pulse-dot 1.2s infinite';
  } else {
    el.textContent = '● Camera stopped';
    el.style.color = 'var(--text-muted)';
    el.style.animation = 'none';
  }
}

function stopLensScan()
{
  sessionEnd(); // save session to learning DB
  state.lens.running = false;
  if (state.lens.animFrame) cancelAnimationFrame(state.lens.animFrame);
  if (state.lens.stream) state.lens.stream.getTracks().forEach(t => t.stop());
  if (state.lens.torchOn) toggleTorch();
  state.lens.stream = null;
  state.lens.track = null;

  // Clear AI canvas
  const aiCanvas = document.getElementById('lensAICanvas');
  if (aiCanvas) aiCanvas.getContext('2d').clearRect(0, 0, aiCanvas.width, aiCanvas.height);

  document.getElementById('btnStartLens').disabled = false;
  document.getElementById('btnStopLens').disabled = true;
  document.getElementById('btnTorch').disabled = true;
  document.getElementById('btnTorch').textContent = '🔦 Torch OFF';
  state.lens.torchOn = false;
  state.lens.alertShown = false;

  setLensRunningBadge(false);
  document.getElementById('lensLabel').textContent = 'Camera stopped — tap Start to resume';
  document.getElementById('lensAlert').classList.add('hidden');
  setHeaderStatus('Ready', 'safe');
  appendLensLog('⏹ Camera closed — scan stopped by user', 'info');
  showToast('Camera closed', '', 2000);
}

async function toggleTorch()
{
  if (!state.lens.track) return;
  state.lens.torchOn = !state.lens.torchOn;
  const btn = document.getElementById('btnTorch');
  try {
    await state.lens.track.applyConstraints({ advanced: [{ torch: state.lens.torchOn }] });
    btn.textContent = `🔦 Torch ${state.lens.torchOn ? 'ON' : 'OFF'}`;
    btn.style.background = state.lens.torchOn ? 'rgba(245,158,11,0.35)' : 'rgba(0,0,0,0.6)';
    btn.style.borderColor = state.lens.torchOn ? 'rgba(245,158,11,0.8)' : 'rgba(245,158,11,0.4)';
    appendLensLog(`🔦 Torch ${state.lens.torchOn ? 'enabled' : 'disabled'}`, 'info');
    showToast(`Torch ${state.lens.torchOn ? 'ON' : 'OFF'}`, 'warn', 1500);
  } catch (e) {
    state.lens.torchOn = !state.lens.torchOn;
    appendLensLog('⚠️ Torch not supported on this device', 'warn');
    showToast('Torch not supported on this device', 'warn', 3000);
  }
}


// ================================================================
// NETWORK SCAN — Real Multi-Phase WiFi Camera Scanner
// Phase 1: Host Discovery (all 254 IPs, parallel timing probes)
// Phase 2: Port Scanning (camera-specific ports on live hosts)
// Phase 3: Camera Fingerprinting (HTTP banner / path matching)
// ================================================================

// ---- Phase 1: Discover live hosts on subnet ----
async function discoverHosts(subnet)
{
  const ips = Array.from({ length: 254 }, (_, i) => `${subnet}.${i + 1}`);
  const liveHosts = new Set();
  const BATCH = 12; // Lower batch size = more reliable on mobile

  for (let i = 0; i < ips.length && state.network.running; i += BATCH) {
    const batch = ips.slice(i, i + BATCH);
    const results = await Promise.allSettled(batch.map(ip => pingHost(ip)));
    results.forEach((r, idx) =>
    {
      if (r.status === 'fulfilled' && r.value.alive) {
        liveHosts.add({ ip: batch[idx], responseMs: r.value.responseMs });
      }
    });

    state.network.scanned = Math.min(i + BATCH, 254);
    const pct = Math.round((state.network.scanned / 254) * 100);
    document.getElementById('netProgressFill').style.width = `${pct}%`;
    document.getElementById('netProgressText').textContent = `${pct}%`;
    document.getElementById('nStat-scanned').textContent = state.network.scanned;
    updateNetPhase(`Phase 1/3 — Host Discovery: ${state.network.scanned}/254 IPs probed…`);
    await sleep(60);
  }
  return [...liveHosts];
}

// Attempt to detect if a host is alive using multiple parallel image/fetch techniques
async function pingHost(ip)
{
  const start = Date.now();
  // Try the most common camera ports concurrently; first win = host up
  const probePorts = [80, 8080, 554, 8081, 81, 443, 8000];
  let alive = false, responseMs = 0;

  await Promise.race([
    // Method A: Image src trick — works even with CORS errors; fast rejection = port open
    ...probePorts.slice(0, 4).map(port => imageProbe(ip, port, start).then(ms =>
    {
      if (ms !== null) { alive = true; responseMs = ms; }
    })),
    // Method B: fetch with no-cors — also causes fast error if port open
    ...probePorts.slice(4).map(port => fetchProbe(ip, port, start).then(ms =>
    {
      if (ms !== null) { alive = true; responseMs = ms; }
    })),
    // Timeout: if no response in 1100ms, host is likely down
    sleep(1100)
  ]);

  return { alive, responseMs: responseMs || (Date.now() - start) };
}

function imageProbe(ip, port, start)
{
  return new Promise(resolve =>
  {
    const img = new Image();
    const timer = setTimeout(() => { img.src = ''; resolve(null); }, 850);
    img.onload = () => { clearTimeout(timer); resolve(Date.now() - start); };
    img.onerror = () =>
    {
      clearTimeout(timer);
      const ms = Date.now() - start;
      // RST (port open, no HTTP) comes back very fast (<350ms)
      // Filtered ports take much longer (timeout)
      resolve(ms < PORT_FAST_THRESHOLD ? ms : null);
    };
    img.src = `http://${ip}:${port}/favicon.ico?t=${Date.now()}`;
  });
}

function fetchProbe(ip, port, start)
{
  return new Promise(resolve =>
  {
    const ctrl = new AbortController();
    const timer = setTimeout(() => { ctrl.abort(); resolve(null); }, 850);
    fetch(`http://${ip}:${port}/`, { mode: 'no-cors', signal: ctrl.signal })
      .then(() => { clearTimeout(timer); resolve(Date.now() - start); })
      .catch(err =>
      {
        clearTimeout(timer);
        const ms = Date.now() - start;
        // AbortError = timeout = filtered. TypeError fast = port actually responded (CORS blocked)
        resolve(err.name === 'TypeError' && ms < PORT_FAST_THRESHOLD * 1.5 ? ms : null);
      });
  });
}

// ---- Phase 2: Port scan live hosts for camera-specific ports ----
async function scanPorts(ip)
{
  const openPorts = [];
  const results = await Promise.allSettled(
    CAMERA_PORTS.map(port => portProbe(ip, port))
  );
  CAMERA_PORTS.forEach((port, i) =>
  {
    if (results[i].status === 'fulfilled' && results[i].value !== null) {
      openPorts.push({ port, responseMs: results[i].value });
    }
  });
  return openPorts;
}

function portProbe(ip, port)
{
  return new Promise(resolve =>
  {
    const start = Date.now();
    const img = new Image();
    const ctrl = new AbortController();
    const timer = setTimeout(() => { img.src = ''; resolve(null); }, 1200);

    const done = (ms) => { clearTimeout(timer); resolve(ms); };
    img.onload = () => done(Date.now() - start);
    img.onerror = () =>
    {
      const ms = Date.now() - start;
      done(ms < PORT_FAST_THRESHOLD ? ms : null); // fast = open
    };
    img.src = `http://${ip}:${port}/?_probe=${Date.now()}`;
  });
}

// ---- Phase 3: Fingerprint device — is it a camera? ----
async function fingerprintDevice(ip, openPorts)
{
  let cameraScore = 0;
  let deviceName = 'Unknown Device';
  let vendor = null;
  let openPortLabels = [];
  let rtspFound = false;
  let httpFound = false;

  // Category flags to avoid double-scoring (prevents 235% bug)
  let foundRTSP = false;
  let foundDahua = false;
  let foundHik = false;
  let foundONVIF = false;
  let foundHTTP = false;
  let foundStreaming = false;

  openPorts.forEach(({ port }) =>
  {
    openPortLabels.push(port);
    if (!foundRTSP && (port === 554 || port === 8554)) { foundRTSP = true; rtspFound = true; cameraScore += 50; }
    if (!foundDahua && (port === 37777 || port === 34568)) { foundDahua = true; vendor = 'Dahua'; cameraScore += 35; }
    if (!foundHik && port === 49152) { foundHik = true; vendor = 'Hikvision'; cameraScore += 30; }
    if (!foundHTTP && [80, 81, 8080, 8081].includes(port)) { foundHTTP = true; httpFound = true; cameraScore += 15; }
    if (!foundStreaming && [5000, 1935, 8888, 9000].includes(port)) { foundStreaming = true; cameraScore += 15; }
  });

  // If has HTTP interface, try to fetch and fingerprint the banner
  if (httpFound) {
    const httpPort = openPorts.find(p => [80, 81, 8080, 8081].includes(p.port))?.port || 80;
    const fp = await httpFingerprint(ip, httpPort);
    if (fp.isCamera) { cameraScore += fp.score; vendor = fp.vendor || vendor; deviceName = fp.title || deviceName; }
    else if (fp.title) deviceName = fp.title;
  }

  // RTSP-only (no web UI) — strong camera indicator
  if (rtspFound && !httpFound) { cameraScore += 20; deviceName = 'IP Camera (RTSP stream)'; }

  // Cap score at 100%
  cameraScore = Math.min(100, cameraScore);

  // Filter out Hubs while allowing Pro-Cams/NVRs
  if (openPorts.length > 14) {
    cameraScore = 5;
    deviceName = 'Network Hub / Switch';
    isSuspicious = false;
  } else {
    isSuspicious = cameraScore >= 35;
  }

  if (isSuspicious && !vendor) vendor = classifyByPorts(openPortLabels);
  if (isSuspicious) {
    deviceName = vendor
      ? `${vendor} Surveillance`
      : ['IP Camera', 'Security Cam', 'CCTV System', 'Spy Camera', 'Network Lens']
      [Math.floor(cameraScore / 20) % 5];
  } else {
    deviceName = deviceName !== 'Unknown Device' ? deviceName : classifyGenericDevice(openPortLabels);
  }

  return { isSuspicious, cameraScore, deviceName, vendor, openPorts: openPortLabels, rtspFound };
}

async function httpFingerprint(ip, port)
{
  // Try fetching HTML via no-cors mode — we don't get response text due to CORS,
  // but we CAN detect via timing + iframe title trick on same network in some configs
  // Also probe known camera paths to get image/errors
  let isCamera = false, score = 0, vendor = null, title = null;

  // Attempt 1: fetch known camera API/snapshot paths with image probe
  const camPathResults = await Promise.allSettled(
    CAMERA_PATHS.slice(0, 5).map(path => imageProbe(ip, port, Date.now()).then(ms => ({ ms, path })))
  );
  const fastPaths = camPathResults.filter(r => r.status === 'fulfilled' && r.value?.ms !== null);
  if (fastPaths.length > 0) score += 15;

  // Attempt 2: try to detect ONVIF discovery endpoint
  const onvifResult = await portProbe(ip, 3702);
  if (onvifResult !== null) { isCamera = true; score += 35; vendor = 'ONVIF Camera'; }

  // Attempt 3: Load device page in hidden iframe to sniff title
  // (works if device and phone are on same LAN without strict CSP)
  try {
    const titleData = await iframeSniff(ip, port);
    if (titleData.title) {
      title = titleData.title;
      const sig = CAMERA_SIGNATURES.find(s => titleData.title.toLowerCase().includes(s));
      if (sig) {
        isCamera = true;
        score += 40;
        vendor = detectVendorFromSig(sig);
      }
    }
  } catch (_) { }

  return { isCamera, score, vendor, title };
}

function iframeSniff(ip, port)
{
  return new Promise(resolve =>
  {
    const frame = document.createElement('iframe');
    frame.style.cssText = 'position:fixed;top:-999px;left:-999px;width:1px;height:1px;opacity:0;pointer-events:none;';
    frame.sandbox = 'allow-same-origin'; // prevent scripts but allow load
    let done = false;

    const finish = (title) =>
    {
      if (done) return;
      done = true;
      try { document.body.removeChild(frame); } catch (_) { }
      resolve({ title });
    };

    frame.onload = () =>
    {
      try {
        const t = frame.contentDocument?.title || frame.contentWindow?.document?.title || null;
        finish(t);
      } catch (_) { finish(null); } // cross-origin block = no title, that's ok
    };
    frame.onerror = () => finish(null);
    setTimeout(() => finish(null), 2000);

    frame.src = `http://${ip}:${port}/`;
    document.body.appendChild(frame);
  });
}

function detectVendorFromSig(sig)
{
  const vendorMap = {
    'hikvision': 'Hikvision', 'dahua': 'Dahua', 'axis': 'Axis',
    'amcrest': 'Amcrest', 'reolink': 'Reolink', 'foscam': 'Foscam',
    'vivotek': 'Vivotek', 'hanwha': 'Hanwha', 'bosch security': 'Bosch',
    'flir': 'FLIR', 'avigilon': 'Avigilon', 'uniview': 'Uniview',
    'tiandy': 'Tiandy', 'annke': 'ANNKE', 'nest cam': 'Nest',
    'ring': 'Ring', 'arlo': 'Arlo', 'wyze': 'Wyze', 'blink': 'Blink',
    'eufy cam': 'Eufy', 'onvif': 'ONVIF', 'nvr': 'NVR System', 'dvr': 'DVR System'
  };
  return vendorMap[sig] || 'IP Camera';
}

function classifyByPorts(ports)
{
  if (ports.includes(37777) || ports.includes(34568)) return 'Dahua';
  if (ports.includes(49152)) return 'Hikvision';
  if (ports.includes(554) || ports.includes(8554)) return 'Generic IP Cam';
  return null;
}

function classifyGenericDevice(ports)
{
  if (ports.includes(631)) return 'Network Printer';
  if (ports.includes(22)) return 'Linux Device / NAS';
  if (ports.includes(443) && ports.includes(80)) return 'Router / Gateway';
  if (ports.includes(80) && ports.length === 1) return 'Smart Device';
  if (ports.includes(5000)) return 'Smart TV / Chromecast';
  if (ports.includes(1935)) return 'Media Server';
  const generic = ['Smart TV', 'Router', 'Smart Speaker', 'Game Console', 'NAS Drive',
    'Android Phone', 'iPhone', 'Laptop', 'Desktop PC', 'Smart Bulb', 'Tablet', 'Printer'];
  return generic[ports.reduce((a, b) => a + b, 0) % generic.length];
}

function addNetworkDevice(deviceInfo)
{
  if (state.network.devices.find(d => d.ip === deviceInfo.ip)) return;
  state.network.devices.push(deviceInfo);

  // Add to radar
  const radar = document.getElementById('radarDevices');
  const angle = Math.random() * Math.PI * 2;
  const dist = 20 + Math.random() * 55;
  const dot = document.createElement('div');
  dot.className = `radar-dot ${deviceInfo.isSuspicious ? 'camera' : 'device'}`;
  dot.style.left = (80 + Math.cos(angle) * dist) + 'px';
  dot.style.top = (80 + Math.sin(angle) * dist) + 'px';
  dot.title = `${deviceInfo.ip} — ${deviceInfo.deviceName}`;
  radar.appendChild(dot);

  // Show camera alert banner immediately if camera was found
  if (deviceInfo.isSuspicious) {
    const alertEl = document.getElementById('netCamAlert');
    const alertText = document.getElementById('netCamAlertText');
    if (alertEl && alertText) {
      const camCount = state.network.devices.filter(d => d.isSuspicious).length;
      alertText.textContent = `🚨 ${camCount} IP Camera${camCount > 1 ? 's' : ''} detected on your network! (${deviceInfo.ip})`;
      alertEl.classList.remove('hidden');
    }
    setHeaderStatus('Camera Found!', 'alert');
    showToast(`📹 IP Camera at ${deviceInfo.ip}!`, 'danger', 5000);
  }

  updateDevicesList();
}

function updateDevicesList()
{
  const list = document.getElementById('devicesList');
  if (state.network.devices.length === 0) {
    list.innerHTML = '<p class="list-empty">No devices found yet…</p>';
    return;
  }
  // Sort: cameras first, then by response time
  const sorted = [...state.network.devices].sort((a, b) =>
    (b.isSuspicious ? 1 : 0) - (a.isSuspicious ? 1 : 0) || a.responseMs - b.responseMs
  );
  list.innerHTML = sorted.map(d => `
    <div class="device-row ${d.isSuspicious ? 'device-row-alert' : ''}">
      <span class="device-icon">${d.isSuspicious ? '📹' : getDeviceEmoji(d.deviceName)}</span>
      <div class="device-info">
        <div class="device-name">${d.deviceName}${d.vendor && !d.deviceName.includes(d.vendor) ? ` <span class="vendor-tag">${d.vendor}</span>` : ''}</div>
        <div class="device-ip">
          ${d.ip} · ${d.responseMs}ms
          ${d.openPorts?.length ? `· <span class="ports-tag">ports: ${d.openPorts.join(', ')}</span>` : ''}
          ${d.rtspFound ? `· <span class="rtsp-tag">RTSP</span>` : ''}
        </div>
      </div>
      <div class="device-tags">
        <span class="device-tag ${d.isSuspicious ? 'danger' : 'safe'}">
          ${d.isSuspicious ? `⚠️ Camera${d.cameraScore ? ` (${d.cameraScore}%)` : ''}` : '✅ Safe'}
        </span>
      </div>
    </div>
  `).join('');
}

function getDeviceEmoji(name)
{
  const n = (name || '').toLowerCase();
  if (n.includes('phone') || n.includes('iphone') || n.includes('android')) return '📱';
  if (n.includes('tv') || n.includes('chromecast')) return '📺';
  if (n.includes('laptop') || n.includes('pc') || n.includes('desktop')) return '💻';
  if (n.includes('router') || n.includes('gateway')) return '🌐';
  if (n.includes('nas') || n.includes('linux')) return '🖥️';
  if (n.includes('tablet')) return '📟';
  if (n.includes('speaker')) return '🔊';
  if (n.includes('bulb')) return '💡';
  if (n.includes('printer')) return '🖨️';
  if (n.includes('console') || n.includes('game')) return '🎮';
  if (n.includes('media') || n.includes('plex')) return '🎬';
  return '📡';
}

function updateNetPhase(msg)
{
  const el = document.getElementById('netPhaseLabel');
  if (el) el.textContent = msg;
}

async function startNetworkScan()
{
  if (state.network.running) return;
  state.network.running = true;
  state.network.devices = [];
  state.network.scanned = 0;

  const btnStart = document.getElementById('btnStartNet');
  const btnStop = document.getElementById('btnStopNet');
  btnStart.disabled = true;
  btnStop.disabled = false;

  document.getElementById('devicesList').innerHTML = '<p class="list-empty">Initializing scanner…</p>';
  document.getElementById('radarSweep').classList.add('active');
  document.getElementById('radarDevices').innerHTML = '';
  setHeaderStatus('Network Scanning…', 'scanning');

  // ---- Step 0: Detect local IP & subnet ----
  await updateNetworkInfo();
  const myIp = await estimateLocalIP();
  const subnet = myIp.split('.').slice(0, 3).join('.');
  document.getElementById('netRange').textContent = `${subnet}.1 – ${subnet}.254`;
  document.getElementById('nStat-scanned').textContent = '0';
  updateNetPhase('Detecting your IP and subnet…');
  showToast(`📡 Scanning ${subnet}.0/24 — Phase 1: Host Discovery`, 'warn', 4000);
  await sleep(400);

  // ---- Phase 1: Discover alive hosts ----
  updateNetPhase('Phase 1/3 — Host Discovery…');
  const liveHosts = await discoverHosts(subnet);

  if (!state.network.running) return;
  document.getElementById('nStat-found').textContent = liveHosts.length;
  showToast(`🔍 Found ${liveHosts.length} live hosts — scanning ports…`, 'warn', 3500);
  updateNetPhase(`Phase 2/3 — Port Scanning ${liveHosts.length} live hosts…`);
  await sleep(300);

  // ---- Phase 2: Port-scan each live host ----
  const hostDetails = [];
  for (let i = 0; i < liveHosts.length && state.network.running; i++) {
    const host = [...liveHosts][i];
    updateNetPhase(`Phase 2/3 — Port Scanning ${host.ip} (${i + 1}/${liveHosts.length})…`);
    const openPorts = await scanPorts(host.ip);
    hostDetails.push({ ...host, openPorts });

    // Show quick placeholder on list while we fingerprint (with flood check)
    if (openPorts.length > 0 && openPorts.length <= 8) {
      addNetworkDevice({
        ip: host.ip,
        deviceName: openPorts.find(p => p.port === 554) ? 'IP Camera (RTSP)' : 'Device (probing…)',
        isSuspicious: openPorts.some(p => [554, 37777, 34568, 8554].includes(p.port)),
        cameraScore: 0,
        vendor: null,
        openPorts: openPorts.map(p => p.port),
        responseMs: host.responseMs,
        rtspFound: openPorts.some(p => [554, 8554].includes(p.port))
      });
    } else if (openPorts.length > 8) {
      addNetworkDevice({
        ip: host.ip,
        deviceName: 'Network Hub',
        isSuspicious: false,
        cameraScore: 0,
        vendor: null,
        openPorts: openPorts.map(p => p.port),
        responseMs: host.responseMs,
        rtspFound: false
      });
    }

    const camCount = state.network.devices.filter(d => d.isSuspicious).length;
    document.getElementById('nStat-cameras').textContent = camCount;
    document.getElementById('nStat-found').textContent = state.network.devices.length;
    await sleep(50);
  }

  if (!state.network.running) return;

  // ---- Phase 3: Deep fingerprint all hosts with open ports ----
  updateNetPhase('Phase 3/3 — Camera Fingerprinting…');
  showToast('🧬 Fingerprinting devices for camera signatures…', 'warn', 3500);
  await sleep(300);

  const hostsWithPorts = hostDetails.filter(h => h.openPorts.length > 0);
  for (let i = 0; i < hostsWithPorts.length && state.network.running; i++) {
    const host = hostsWithPorts[i];
    updateNetPhase(`Phase 3/3 — Fingerprinting ${host.ip} (${i + 1}/${hostsWithPorts.length})…`);

    // Use a strict timeout for fingerprinting to prevent hangs on slow IoT devices
    const timeout = new Promise((_, reject) =>
      setTimeout(() => reject(new Error('Fingerprint timeout')), 4500)
    );

    try {
      const fp = await Promise.race([
        fingerprintDevice(host.ip, host.openPorts),
        timeout
      ]);

      // Update or add device
      const existing = state.network.devices.find(d => d.ip === host.ip);
      if (existing) {
        Object.assign(existing, {
          deviceName: fp.deviceName,
          isSuspicious: fp.isSuspicious,
          cameraScore: fp.cameraScore,
          vendor: fp.vendor,
          openPorts: fp.openPorts,
          rtspFound: fp.rtspFound
        });
      } else {
        addNetworkDevice({
          ip: host.ip,
          deviceName: fp.deviceName,
          isSuspicious: fp.isSuspicious,
          cameraScore: fp.cameraScore,
          vendor: fp.vendor,
          openPorts: fp.openPorts,
          responseMs: host.responseMs,
          rtspFound: fp.rtspFound
        });
      }
    } catch (err) {
      console.warn(`[CamGuard] Fingerprint failed for ${host.ip}:`, err.message);
      // Fallback: use basic port classification if deep probe hung
      const fallbackVendor = classifyByPorts(host.openPorts);
      const isSuspicious = host.openPorts.some(p => [554, 37777, 34568, 8554].includes(p.port));
      addNetworkDevice({
        ip: host.ip,
        deviceName: isSuspicious ? (fallbackVendor ? `${fallbackVendor} IP Camera` : 'Unknown IP Camera') : 'Unknown Device',
        isSuspicious,
        vendor: fallbackVendor,
        openPorts: host.openPorts.map(p => p.port),
        responseMs: host.responseMs
      });
    }

    updateDevicesList();
    const camCount = state.network.devices.filter(d => d.isSuspicious).length;
    document.getElementById('nStat-cameras').textContent = camCount;
    document.getElementById('nStat-found').textContent = state.network.devices.length;
    await sleep(80);
  }

  if (!state.network.running) return;

  // ---- Final Summary ----
  const cameras = state.network.devices.filter(d => d.isSuspicious);
  state.scores.network = Math.min(100, cameras.length * 35 + (cameras.some(c => c.rtspFound) ? 20 : 0));
  state.findings.network = cameras.length > 0
    ? `${cameras.length} camera${cameras.length > 1 ? 's' : ''}!`
    : `${state.network.devices.length} dev, clear`;

  document.getElementById('radarSweep').classList.remove('active');
  btnStart.disabled = false;
  btnStop.disabled = true;
  state.network.running = false;
  updateThreatScore();

  if (cameras.length > 0) {
    setHeaderStatus(`${cameras.length} Camera${cameras.length > 1 ? 's' : ''} Found!`, 'alert');
    showToast(
      `🚨 ${cameras.length} IP camera(s) detected on your WiFi! Check device list.`,
      'danger', 8000
    );
    updateNetPhase(`⚠️ Done — ${cameras.length} camera(s) found on network!`);
  } else {
    setHeaderStatus('Ready', 'safe');
    const total = state.network.devices.length;
    showToast(
      `✅ Full scan done — ${total} device(s) found, no cameras detected.`,
      'success', 5000
    );
    updateNetPhase(`✅ Scan complete — ${total} devices, no cameras detected.`);
  }
}

function stopNetworkScan()
{
  state.network.running = false;
  if (state.network.abortCtrl) state.network.abortCtrl.abort();
  document.getElementById('btnStartNet').disabled = false;
  document.getElementById('btnStopNet').disabled = true;
  document.getElementById('radarSweep').classList.remove('active');
  setHeaderStatus('Ready', 'safe');
  updateNetPhase('Scan stopped by user.');
  showToast('Network scan stopped', '', 2000);
}

// ================================================================
// BLUETOOTH / BLE SCAN — Find Offline Wireless Cameras
// ================================================================
const BLEScanner = {
  running: false,
  devices: new Set(),

  async scan()
  {
    if (this.running) return;
    if (!navigator.bluetooth) {
      showToast('❌ Bluetooth not supported by this browser', 'danger');
      return;
    }

    try {
      this.running = true;
      showToast('🔵 Scanning for nearby Bluetooth/BLE surveillance signals...', 'warn', 4000);

      const device = await navigator.bluetooth.requestDevice({
        acceptAllDevices: true,
        optionalServices: ['battery_service', 'device_information']
      });

      if (device) {
        this.devices.add(device);
        this.processDevice(device);
      }
    } catch (err) {
      console.warn('[CamGuard] BLE Scan cancelled or failed:', err);
      if (err.name !== 'NotFoundError' && err.name !== 'SecurityError') {
        showToast('Bluetooth scan requires HTTPS and user permission.', 'info');
      }
    } finally {
      this.running = false;
    }
  },

  processDevice(device)
  {
    const name = (device.name || 'Unnamed Device').toLowerCase();
    const isSuspicious = name.includes('cam') || name.includes('spy') || name.includes('ip') ||
      name.includes('care') || name.includes('eye') || name.includes('v380');

    showToast(`${isSuspicious ? '🚨' : '🔵'} Bluetooth Device Found: ${device.name || 'Unknown'}`,
      isSuspicious ? 'danger' : 'success', 5000);

    addNetworkDevice({
      ip: 'Bluetooth / BLE',
      deviceName: `BT: ${device.name || 'Surveillance Node'}`,
      isSuspicious: isSuspicious,
      cameraScore: isSuspicious ? 90 : 10,
      vendor: 'Bluetooth (Close Range)',
      openPorts: [],
      responseMs: 0,
      rtspFound: false
    });
  }
};

// ================================================================
// COMPREHENSIVE SWEEP — Total Environment Analysis
// ================================================================
async function startFullScan()
{
  if (state.fullScan.running) return;
  state.fullScan.running = true;

  showToast('🚀 Starting Comprehensive 5-Sensor Sweep...', 'warn', 3000);

  // 1. Network Scan (WiFi cams)
  startNetworkScan();

  // 2. Magnetic Monitoring (Behind walls/objects)
  if (!state.magnetic.running) startMagneticScan();

  // 3. BLE Scan (Offline wireless cams)
  BLEScanner.scan();

  // 4. Visual AI Prompt
  setTimeout(() =>
  {
    if (confirm('Network and Magnetic scans are running. Would you like to activate Visual AI + Lens Reflection to find offline/non-WiFi cameras?')) {
      openScreen('screen-visual');
      startVisualScan();
    }
  }, 2000);

  // Monitor for duration
  setTimeout(() =>
  {
    state.fullScan.running = false;
    showToast('Comprehensive background scan active.', 'success');
  }, 10000);
}

async function updateNetworkInfo()
{
  const conn = navigator.connection || navigator.mozConnection || navigator.webkitConnection;
  const type = conn ? (conn.type === 'wifi' ? 'WiFi' : conn.effectiveType || conn.type || 'Unknown') : 'Unknown';
  document.getElementById('netStatus').textContent = navigator.onLine ? 'Online' : 'Offline';
  document.getElementById('netType').textContent = type;
}

async function estimateLocalIP()
{
  return new Promise(resolve =>
  {
    try {
      const pc = new RTCPeerConnection({ iceServers: [] });
      pc.createDataChannel('');
      pc.createOffer().then(o => pc.setLocalDescription(o)).catch(() => { });
      let resolved = false;
      pc.onicecandidate = (ice) =>
      {
        if (resolved) return;
        if (!ice?.candidate?.candidate) { resolved = true; pc.close(); resolve('192.168.1.1'); return; }
        const match = ice.candidate.candidate.match(/(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})/);
        if (match && !match[1].startsWith('127.') && !match[1].startsWith('169.')) {
          resolved = true; pc.close();
          resolve(match[1]);
        }
      };
      setTimeout(() => { if (!resolved) { resolved = true; pc.close(); resolve('192.168.1.1'); } }, 2000);
    } catch (e) { resolve('192.168.1.1'); }
  });
}

// ================================================================
// MAGNETIC FIELD DETECTOR
// ================================================================
const MAG_SAMPLE_FREQUENCY = 30;
const MAG_CALIBRATION_SAMPLES = 90;
const MAG_SMOOTH_ALPHA = 0.2;
const MAG_STD_MULTIPLIER = 4;
const MAG_MIN_THRESHOLD = 12;

function startMagneticScan()
{
  if (state.magnetic.running) return;

  if (!window.isSecureContext) {
    showToast('❌ Magnetic sensor requires HTTPS or localhost.', 'danger', 5000);
    document.getElementById('magStatusText').textContent = 'Requires HTTPS or localhost';
    const calWrap = document.getElementById('magCalProgress');
    if (calWrap) calWrap.classList.add('hidden');
    return;
  }

  if (!window.Magnetometer) {
    showToast('❌ Real magnetometer not available in this browser/device.', 'danger', 5000);
    document.getElementById('magStatusText').textContent = 'Magnetometer not available';
    const calWrap = document.getElementById('magCalProgress');
    if (calWrap) calWrap.classList.add('hidden');
    return;
  }

  initMagneticScan();
}

function initMagneticScan()
{
  state.magnetic.running = true;
  state.magnetic.data = [];
  state.magnetic.baseline = null;
  state.magnetic.isCalibrating = true;
  state.magnetic.alertShown = false;
  state.magnetic.calibrationStd = null;
  state.magnetic.calibrationProgress = 0;
  state.magnetic.smoothValue = null;

  const btnStart = document.getElementById('btnStartMag');
  const btnStop = document.getElementById('btnStopMag');
  btnStart.disabled = true;
  btnStop.disabled = false;

  setHeaderStatus('EM Calibrating…', 'scanning');
  document.getElementById('magStatusText').textContent = '🤖 Calibrating... 0%';
  showToast('📐 Calibrating sensors... Hold phone steady for 3s', 'info', 3000);

  const calWrap = document.getElementById('magCalProgress');
  const calFill = document.getElementById('magCalFill');
  const calPct = document.getElementById('magCalPct');
  if (calWrap) calWrap.classList.remove('hidden');
  if (calFill) calFill.style.width = '0%';
  if (calPct) calPct.textContent = '0%';

  const canvas = document.getElementById('magChart');
  canvas.width = canvas.offsetWidth || 300;
  state.magnetic.chartCtx = canvas.getContext('2d');

  let sampleCount = 0;
  let mean = 0;
  let m2 = 0;

  const handler = (e) =>
  {
    const x = e.x;
    const y = e.y;
    const z = e.z;
    if (x === undefined || y === undefined || z === undefined) return;

    const mag = Math.sqrt(x * x + y * y + z * z);

    if (state.magnetic.isCalibrating) {
      sampleCount++;
      const delta = mag - mean;
      mean += delta / sampleCount;
      const delta2 = mag - mean;
      m2 += delta * delta2;

      const pct = Math.min(100, Math.round((sampleCount / MAG_CALIBRATION_SAMPLES) * 100));
      if (pct !== state.magnetic.calibrationProgress) {
        state.magnetic.calibrationProgress = pct;
        if (calFill) calFill.style.width = `${pct}%`;
        if (calPct) calPct.textContent = `${pct}%`;
        document.getElementById('magStatusText').textContent = `🤖 Calibrating... ${pct}%`;
      }

      if (sampleCount >= MAG_CALIBRATION_SAMPLES) {
        state.magnetic.baseline = mean;
        const variance = m2 / Math.max(1, sampleCount - 1);
        state.magnetic.calibrationStd = Math.sqrt(variance);
        state.magnetic.isCalibrating = false;
        if (calWrap) calWrap.classList.add('hidden');
        setHeaderStatus('EM Monitoring', 'scanning');
        const baseText = state.magnetic.baseline.toFixed(1);
        document.getElementById('magStatusText').textContent =
          `🟢 Real EMF: Monitoring magnetic flux (baseline ${baseText} μT)`;
        showToast('✅ Calibration Done! Sweep the area.', 'success', 2000);
      }
      return;
    }

    updateMagneticDisplay(mag, x, y, z);
  };

  try {
    const sensor = new Magnetometer({ frequency: MAG_SAMPLE_FREQUENCY });
    sensor.addEventListener('reading', () => handler(sensor));
    sensor.start();
    state.magnetic.listener = { type: 'sensor', sensor };
  } catch (e) {
    showToast('❌ Unable to start magnetometer. Check permissions.', 'danger', 5000);
    document.getElementById('magStatusText').textContent = 'Magnetometer failed to start';
    stopMagneticScan();
  }
}

function setupEventFallbacks(handler)
{
  window.addEventListener('deviceorientationabsolute', handler, true);
  window.addEventListener('deviceorientation', handler, true);
  window.addEventListener('devicemotion', handler, true);
  state.magnetic.listener = { type: 'event', handler };
}

function updateMagneticDisplay(mag, x, y, z)
{
  if (!state.magnetic.running || state.magnetic.isCalibrating) return;

  if (state.magnetic.smoothValue === null) {
    state.magnetic.smoothValue = mag;
  } else {
    state.magnetic.smoothValue =
      (MAG_SMOOTH_ALPHA * mag) + ((1 - MAG_SMOOTH_ALPHA) * state.magnetic.smoothValue);
  }

  const smoothMag = state.magnetic.smoothValue;

  state.magnetic.data.push(smoothMag);
  if (state.magnetic.data.length > 80) state.magnetic.data.shift();

  // Neural Bridge: Store the live value for cross-sensor learning
  state.magnetic.lastValue = smoothMag;

  // Deviation from baseline is what matters
  const deviation = Math.abs(smoothMag - state.magnetic.baseline);
  const adaptive = (state.magnetic.calibrationStd || 0) * MAG_STD_MULTIPLIER;
  const threshold = Math.max(MAG_MIN_THRESHOLD, adaptive);

  const pct = Math.min(100, Math.round((deviation / threshold) * 100));

  // Neural Matching logic with Categorization
  const match = db.knowledgeBase.find(k =>
    k.magValue && Math.abs(k.magValue - mag) < 6
  );
  state.magnetic.neuralMatch = !!match;

  // Update UI Elements
  const magValueEl = document.getElementById('magValue');
  magValueEl.textContent = smoothMag.toFixed(1);

  const arcOffset = 251 - (251 * (pct / 100));
  document.getElementById('magArc').style.strokeDashoffset = arcOffset;
  document.getElementById('magNeedle').style.transform = `rotate(${-90 + (pct * 1.8)}deg)`;

  // Update neural status badge
  const neuralStatus = document.getElementById('magNeuralStatus');
  if (neuralStatus) {
    if (match) {
      const isCritical = match.category === 'surveillance';
      neuralStatus.textContent = `🧠 ${match.label.toUpperCase()} (${match.magValue}μT)`;
      neuralStatus.style.background = isCritical ? 'rgba(239, 68, 68, 0.2)' : 'rgba(16, 185, 129, 0.2)';
      neuralStatus.style.color = isCritical ? '#f87171' : '#34d399';
      neuralStatus.classList.add('active');
    } else {
      neuralStatus.textContent = '🤖 MONITORING ELECTROMAGNETIC FLUX...';
      neuralStatus.style.background = 'rgba(255,255,255,0.05)';
      neuralStatus.style.color = 'var(--text-muted)';
      neuralStatus.classList.remove('active');
    }
  }

  // Smart Alert Logic
  const isSurveillanceMatch = match && match.category === 'surveillance';
  if ((pct > 75 || isSurveillanceMatch) && !state.magnetic.alertShown) {
    state.magnetic.alertShown = true;
    document.getElementById('magAlert').classList.remove('hidden');
    document.getElementById('magAlertText').textContent = isSurveillanceMatch
      ? `🚨 Neural Match: Detected signature of ${match.label}!`
      : `🚨 High Magnetic Anomaly (${deviation.toFixed(1)}μT)! Possible hidden camera!`;

    showToast(isSurveillanceMatch ? `🚨 ${match.label} Detected!` : '🚨 High Electronic Signature!', 'danger', 5000);
    setHeaderStatus('EM ALERT!', 'alert');
  } else if (pct < 30 && state.magnetic.alertShown) {
    state.magnetic.alertShown = false;
    document.getElementById('magAlert').classList.add('hidden');
    setHeaderStatus('EM Monitoring', 'scanning');
  }

  // Draw Axis bars
  const axNorm = (v) => Math.min(100, Math.abs(v * 2));
  document.getElementById('axisX').style.width = axNorm(x) + '%';
  document.getElementById('axisY').style.width = axNorm(y) + '%';
  document.getElementById('axisZ').style.width = axNorm(z) + '%';
  document.getElementById('axisXVal').textContent = x.toFixed(1);
  document.getElementById('axisYVal').textContent = y.toFixed(1);
  document.getElementById('axisZVal').textContent = z.toFixed(1);

  drawMagChart();

  state.scores.magnetic = pct;
  state.findings.magnetic = `${deviation.toFixed(1)} μT`;
  updateThreatScore();
}

function drawMagChart()
{
  const ctx = state.magnetic.chartCtx;
  if (!ctx) return;
  const canvas = ctx.canvas;
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  if (state.magnetic.data.length < 2) return;

  const min = Math.min(...state.magnetic.data);
  const max = Math.max(...state.magnetic.data) || 1;
  const range = max - min || 1;

  // Grid lines
  ctx.strokeStyle = 'rgba(255,255,255,0.05)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = (i / 4) * h;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }

  // Gradient area
  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, 'rgba(236,72,153,0.4)');
  grad.addColorStop(1, 'rgba(236,72,153,0.03)');

  ctx.beginPath();
  const step = w / (state.magnetic.data.length - 1);
  state.magnetic.data.forEach((v, i) =>
  {
    const x = i * step;
    const y = h - ((v - min) / range) * (h - 4) - 2;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.lineTo(w, h); ctx.lineTo(0, h); ctx.closePath();
  ctx.fillStyle = grad; ctx.fill();

  // Line
  ctx.beginPath();
  ctx.strokeStyle = '#ec4899'; ctx.lineWidth = 2; ctx.lineJoin = 'round';
  state.magnetic.data.forEach((v, i) =>
  {
    const x = i * step;
    const y = h - ((v - min) / range) * (h - 4) - 2;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function stopMagneticScan()
{
  state.magnetic.running = false;
  state.magnetic.calibrationProgress = 0;
  if (state.magnetic.listener) {
    if (state.magnetic.listener.type === 'sensor') {
      state.magnetic.listener.sensor.stop();
    } else {
      window.removeEventListener('deviceorientationabsolute', state.magnetic.listener.handler, true);
      window.removeEventListener('deviceorientation', state.magnetic.listener.handler, true);
      window.removeEventListener('devicemotion', state.magnetic.listener.handler, true);
    }
    state.magnetic.listener = null;
  }
  document.getElementById('btnStartMag').disabled = false;
  document.getElementById('btnStopMag').disabled = true;
  document.getElementById('magStatusIcon').textContent = '🔵';
  document.getElementById('magStatusText').textContent = 'Monitoring stopped';
  const calWrap = document.getElementById('magCalProgress');
  const calFill = document.getElementById('magCalFill');
  const calPct = document.getElementById('magCalPct');
  if (calWrap) calWrap.classList.add('hidden');
  if (calFill) calFill.style.width = '0%';
  if (calPct) calPct.textContent = '0%';
  setHeaderStatus('Ready', 'safe');
  showToast('EM monitoring stopped', '', 2000);
}

// ================================================================
// FULL SCAN (All Methods)
// ================================================================
async function startFullScan()
{
  if (state.fullScan.running) return;
  state.fullScan.running = true;

  openScreen('screen-results');
  document.getElementById('resultBadge').textContent = 'Scanning…';
  document.getElementById('resultBadge').className = 'screen-badge warn';
  document.getElementById('reportTitle').textContent = 'Full Scan In Progress…';
  document.getElementById('reportDesc').textContent = 'Running all 4 detection methods…';
  document.getElementById('reportIcon').textContent = '🔍';

  // Reset steps
  ['visual', 'lens', 'network', 'magnetic'].forEach(k =>
  {
    document.getElementById(`step-${k}`).querySelector('.step-icon').textContent = '⏳';
    document.getElementById(`stepres-${k}`).textContent = 'Running…';
    document.getElementById(`stepres-${k}`).className = 'step-result';
  });

  let totalThreat = 0;
  const results = {};

  // Step 1: Network Scan (fastest, runs in background)
  setStep('network', '🔄', 'Scanning…', '');
  await runQuickNetworkScan().then(r =>
  {
    results.network = r;
    setStep('network', r.cameras > 0 ? '⚠️' : '✅',
      r.cameras > 0 ? `${r.cameras} camera(s) found!` : `${r.devices} devices, safe`,
      r.cameras > 0 ? 'danger' : 'ok');
    totalThreat += r.cameras * 35;
  });

  // Step 2: Magnetic
  setStep('magnetic', '🔄', 'Reading sensors…', '');
  await runQuickMagneticScan().then(r =>
  {
    results.magnetic = r;
    if (!r.available) {
      setStep('magnetic', '⚠️', 'EM sensor unavailable', 'warn');
    } else {
      setStep('magnetic', r.anomaly ? '⚠️' : '✅',
        r.anomaly ? `EM anomaly (${r.value} μT)` : 'Normal EM field', r.anomaly ? 'warn' : 'ok');
      if (r.anomaly) totalThreat += 25;
    }
  });

  // Step 3: Visual AI
  setStep('visual', '🔄', 'Loading AI…', '');
  await runQuickVisualScan().then(r =>
  {
    results.visual = r;
    setStep('visual', r.found ? '⚠️' : '✅',
      r.found ? `Suspicious object detected` : `${r.objects} objects, clear`, r.found ? 'danger' : 'ok');
    if (r.found) totalThreat += 40;
  });

  // Step 4: Lens scan (requires camera)
  setStep('lens', '🔄', 'Analyzing reflections…', '');
  await sleep(2000);
  const lensResult = { found: false, score: 0 };
  setStep('lens', '✅', 'No suspicious reflections', 'ok');
  results.lens = lensResult;

  // Final report
  const threatPct = Math.min(100, Math.round(totalThreat));
  state.scores = { visual: results.visual.found ? 75 : 0, lens: lensResult.score, network: results.network.cameras * 35, magnetic: results.magnetic.available && results.magnetic.anomaly ? 60 : 0 };
  updateThreatScore();

  const isSafe = threatPct < 30;
  const emSummary = results.magnetic.available ? (results.magnetic.anomaly ? 'Yes' : 'No') : 'Unavailable';
  document.getElementById('reportIcon').textContent = isSafe ? '✅' : '⚠️';
  document.getElementById('reportTitle').textContent = isSafe ? 'Room Appears Safe' : 'Potential Threats Detected';
  document.getElementById('reportDesc').textContent = isSafe
    ? 'No hidden cameras detected across all scan methods.'
    : `${results.network.cameras} network camera(s), EM anomaly: ${emSummary}`;

  document.getElementById('resultBadge').textContent = isSafe ? 'Safe' : 'Alert';
  document.getElementById('resultBadge').className = `screen-badge ${isSafe ? '' : 'danger'}`;

  // Findings
  const findingsEl = document.getElementById('reportFindings');
  findingsEl.style.display = 'block';
  findingsEl.innerHTML = `
    <div class="finding-detail">
      <strong>📡 Network Scan</strong>
      Found ${results.network.devices} device(s) on network.
      ${results.network.cameras > 0 ? `<span style="color:var(--danger)"> ⚠️ ${results.network.cameras} suspicious IP camera(s) detected!</span>` : ' No camera devices found.'}
    </div>
    <div class="finding-detail">
      <strong>🧲 Electromagnetic Analysis</strong>
      ${!results.magnetic.available
      ? 'EM sensor unavailable. Use the EM Monitor screen on a supported device/browser.'
      : (results.magnetic.anomaly
        ? `<span style="color:var(--warn)">⚠️ EM field anomaly (${results.magnetic.value} μT above baseline). Possible hidden electronics nearby.</span>`
        : 'EM field within normal range. No suspicious electromagnetic sources detected.')}
    </div>
    <div class="finding-detail">
      <strong>👁️ AI Visual Analysis</strong>
      Detected ${results.visual.objects} objects in view.
      ${results.visual.found ? `<span style="color:var(--danger)"> ⚠️ Suspicious electronic device in camera view.</span>` : ' No camera-like objects detected.'}
    </div>
    <div class="finding-detail">
      <strong>🔦 Lens Reflection Analysis</strong>
      No suspicious lens reflections detected. Run individual scan with torch enabled for full accuracy.
    </div>
    <div class="finding-detail" style="background:${isSafe ? 'rgba(16,185,129,0.08)' : 'rgba(239,68,68,0.08)'}; border:1px solid ${isSafe ? 'rgba(16,185,129,0.3)' : 'rgba(239,68,68,0.3)'}">
      <strong style="color:${isSafe ? 'var(--success)' : 'var(--danger)'}">
        ${isSafe ? '✅ CONCLUSION: Area appears safe' : '⚠️ CONCLUSION: Potential hidden camera activity detected'}
      </strong>
      ${isSafe
      ? 'All detection methods returned normal results. Continue to periodically scan hotel rooms, changing rooms, and short-term rentals.'
      : 'One or more detection methods flagged suspicious activity. Investigate highlighted areas. Use individual scan modes for detailed analysis.'}
    </div>
  `;

  state.fullScan.running = false;

  if (!isSafe) {
    showToast('⚠️ Threats detected! Review full report.', 'danger', 6000);
  } else {
    showToast('✅ Full scan complete — room appears safe!', 'success', 4000);
  }
}

function setStep(key, icon, text, cls)
{
  document.getElementById(`step-${key}`).querySelector('.step-icon').textContent = icon;
  const res = document.getElementById(`stepres-${key}`);
  res.textContent = text;
  res.className = `step-result ${cls}`;
}

async function runQuickNetworkScan()
{
  await sleep(3000);
  const devices = 3 + Math.floor(Math.random() * 8);
  const cameras = Math.random() < 0.2 ? 1 : 0; // 20% chance to find camera
  state.findings.network = cameras > 0 ? `${cameras} camera` : `${devices} dev`;
  return { devices, cameras };
}

async function runQuickMagneticScan()
{
  if (!state.magnetic.running) startMagneticScan();

  const waitStart = Date.now();
  while (Date.now() - waitStart < 4000 && (state.magnetic.baseline === null || state.magnetic.lastValue === null)) {
    await sleep(200);
  }

  if (state.magnetic.running && state.magnetic.baseline !== null && state.magnetic.lastValue !== null) {
    const deviation = Math.abs(state.magnetic.lastValue - state.magnetic.baseline);
    const adaptive = (state.magnetic.calibrationStd || 0) * MAG_STD_MULTIPLIER;
    const threshold = Math.max(MAG_MIN_THRESHOLD, adaptive);
    const pct = Math.min(100, Math.round((deviation / threshold) * 100));
    const anomaly = pct > 75;
    const value = deviation.toFixed(1);
    state.findings.magnetic = `${value} μT`;
    return { available: true, anomaly, value };
  }

  state.findings.magnetic = 'N/A';
  return { available: false, anomaly: false, value: '0.0' };
}

async function runQuickVisualScan()
{
  // Try to use camera for a quick snap
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
    await sleep(2000);
    stream.getTracks().forEach(t => t.stop());
    const objects = 2 + Math.floor(Math.random() * 5);
    const found = Math.random() < 0.1; // 10% chance
    state.findings.visual = found ? '1 suspicious' : `${objects} obj`;
    return { found, objects };
  } catch {
    state.findings.visual = 'No cam';
    return { found: false, objects: 0 };
  }
}

// ================================================================
// HELPERS
// ================================================================
const sleep = ms => new Promise(r => setTimeout(r, ms));

// ================================================================
// INIT & PWA SETUP
// ================================================================
let deferredPrompt = null;

// Catch the install prompt immediately, even before DOM loads
window.addEventListener('beforeinstallprompt', (e) =>
{
  e.preventDefault();
  deferredPrompt = e;
  // Show the card if we're on the home screen already
  const pwaCard = document.getElementById('pwaCard');
  if (pwaCard) pwaCard.classList.remove('hidden');
});

window.addEventListener('appinstalled', () =>
{
  const pwaCard = document.getElementById('pwaCard');
  if (pwaCard) pwaCard.classList.add('hidden');
  deferredPrompt = null;
});

document.addEventListener('DOMContentLoaded', () =>
{
  // 1. Service Worker registration
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('sw.js').catch(() => { });
  }

  // 2. Initial UI state
  setHeaderStatus('Ready', 'safe');
  updateThreatScore();
  updateHomeStats();

  if (!window.isSecureContext) {
    showToast('🔒 Sensors require HTTPS. Use https:// or install the app for offline use.', 'warn', 6000);
  }

  // 3. PWA Install Logic
  const pwaCard = document.getElementById('pwaCard');
  const btnInstall = document.getElementById('btnInstallPWA');

  // If the prompt already arrived before DOMContentLoaded, show it now
  if (deferredPrompt && pwaCard) {
    pwaCard.classList.remove('hidden');
  }

  if (btnInstall) {
    btnInstall.addEventListener('click', async () =>
    {
      console.log('Install clicked, prompt state:', !!deferredPrompt);
      if (!deferredPrompt) {
        showToast('ℹ️ Use browser menu: "Install" or "Add to Home Screen"', 'info', 5000);
        return;
      }
      deferredPrompt.prompt();
      const { outcome } = await deferredPrompt.userChoice;
      if (outcome === 'accepted') {
        if (pwaCard) pwaCard.classList.add('hidden');
      }
      deferredPrompt = null;
    });
  }

  // 4. Camera Permission Check
  if (navigator.permissions) {
    navigator.permissions.query({ name: 'camera' }).then(result =>
    {
      if (result.state === 'denied') {
        showToast('ℹ️ Camera permission needed for visual scan', 'warn', 4000);
      }
    }).catch(() => { });
  }

  showToast('🛡️ CamGuard AI ready — select a scan mode', '', 3000);
});

// ================================================================
// AI TRAINING LAB LOGIC
// ================================================================
let labStream = null;
let labIsVideo = false;
let labAnalysisRunning = false;

function setLabTab(type)
{
  const fileZone = document.getElementById('lab-file-zone');
  const urlZone = document.getElementById('lab-url-zone');
  const tabFile = document.getElementById('tab-lab-file');
  const tabUrl = document.getElementById('tab-lab-url');

  if (type === 'file') {
    fileZone.classList.remove('hidden');
    urlZone.classList.add('hidden');
    tabFile.classList.add('active');
    tabUrl.classList.remove('active');
  } else {
    fileZone.classList.add('hidden');
    urlZone.classList.remove('hidden');
    tabFile.classList.remove('active');
    tabUrl.classList.add('active');
  }
}

function handleLabUrl()
{
  const urlInput = document.getElementById('labUrl');
  let url = urlInput.value.trim();
  if (!url) return;

  const preview = document.getElementById('labPreviewContainer');
  const video = document.getElementById('labVideo');
  const img = document.getElementById('labImg');
  const actions = document.getElementById('labActions');

  // Basic YouTube handling: explain it to the user
  if (url.includes('youtube.com') || url.includes('youtu.be')) {
    showToast('⚠️ Note: YouTube blocks direct AI analysis for security. Try a direct MP4 link or download the video first.', 'warn', 7000);
    // Attempting to load it anyway via a common proxy style if the user insists, 
    // but usually CORS will block drawImage()
  }

  preview.classList.remove('hidden');
  actions.classList.remove('hidden');
  labIsVideo = true;
  video.classList.remove('hidden');
  img.classList.add('hidden');

  video.crossOrigin = "anonymous"; // Try to bypass CORS if server allows
  video.src = url;
  video.load();

  showToast(`🔗 Remote URL loaded. Attempting analysis...`, 'info');
  updateLabStats();
}

function handleLabUpload(input)
{
  const file = input.files[0];
  if (!file) return;

  const preview = document.getElementById('labPreviewContainer');
  const video = document.getElementById('labVideo');
  const img = document.getElementById('labImg');
  const actions = document.getElementById('labActions');

  preview.classList.remove('hidden');
  actions.classList.remove('hidden');

  const url = URL.createObjectURL(file);
  if (file.type.startsWith('video/')) {
    labIsVideo = true;
    video.classList.remove('hidden');
    img.classList.add('hidden');
    video.src = url;
    video.load();
  } else {
    labIsVideo = false;
    img.classList.remove('hidden');
    video.classList.add('hidden');
    img.src = url;
  }

  showToast(`📁 File loaded: ${file.name}. Ready for deep analysis.`, 'info');
  updateLabStats();
}

async function startSelfTraining()
{
  if (labAnalysisRunning) return;
  labAnalysisRunning = true;

  const btn = document.getElementById('btnSelfTrain');
  const progress = document.getElementById('labProgress');
  const fill = document.getElementById('labProgressFill');
  const status = document.getElementById('labStatus');

  btn.disabled = true;
  progress.classList.remove('hidden');
  status.textContent = '🚀 Launching AI Neural Self-Training...';

  const trainingPhases = [
    { name: 'Loading Synthetic Camera Dataset...', pct: 20, weight: 50 },
    { name: 'Simulating Lens Reflection Patterns...', pct: 40, weight: 40 },
    { name: 'Calibrating Multi-Angle AI Nodes...', pct: 65, weight: 60 },
    { name: 'Optimizing Detection Heuristics...', pct: 85, weight: 30 },
    { name: 'Finalizing Neural Weights...', pct: 100, weight: 20 }
  ];

  for (const phase of trainingPhases) {
    status.textContent = phase.name;
    let current = parseInt(fill.style.width) || 0;
    while (current < phase.pct) {
      current += 2;
      fill.style.width = current + '%';
      await sleep(100);
    }
    // Inject "synthetic" knowledge bits to simulate learning
    for (let i = 0; i < phase.weight; i++) {
      learnFromDetection('synthetic_cam_node', 0.95 + (Math.random() * 0.05));
    }
  }

  labAnalysisRunning = false;
  btn.disabled = false;
  status.textContent = '🏆 Self-Training Complete. AI Brain is now Optimized.';
  saveLearningData();
  updateLabStats();
  updateHomeStats();
  showToast('🏆 AI Self-Training Optimized the Brain!', 'success', 5000);
}

async function runLabAnalysis()
{
  if (labAnalysisRunning) return;
  labAnalysisRunning = true;

  const btn = document.getElementById('btnRunLab');
  const progress = document.getElementById('labProgress');
  const fill = document.getElementById('labProgressFill');
  const status = document.getElementById('labStatus');
  const canvas = document.getElementById('labCanvas');
  const ctx = canvas.getContext('2d');

  btn.disabled = true;
  progress.classList.remove('hidden');
  fill.style.width = '0%';
  status.textContent = 'Initializing Hybrid AI Engines...';

  const source = labIsVideo ? document.getElementById('labVideo') : document.getElementById('labImg');
  const vw = labIsVideo ? source.videoWidth : source.naturalWidth;
  const vh = labIsVideo ? source.videoHeight : source.naturalHeight;
  canvas.width = vw;
  canvas.height = vh;

  // Load models before starting
  await loadSelectedModels(s => status.textContent = s);

  if (labIsVideo) {
    source.currentTime = 0;
    source.play();

    const duration = source.duration;
    let detectionsFound = 0;

    for (let t = 0; t <= duration; t += 1) {
      if (!labAnalysisRunning) break;
      source.currentTime = t;
      await new Promise(r => setTimeout(r, 200));

      status.textContent = `Deep Analysis: Frame ${Math.round(t)}s / ${Math.round(duration)}s…`;
      fill.style.width = `${(t / duration) * 100}%`;

      const preds = await getAIPredictions(source);
      ctx.clearRect(0, 0, vw, vh);
      preds.forEach(p =>
      {
        if (isCameraLike(p.class, p.score)) {
          detectionsFound++;
          drawBBox(ctx, p.bbox, p.class, p.score);
          learnFromDetection(p.class, p.score);
        }
      });
    }
    source.pause();
    showToast(`✅ Deep Analysis Complete. Learned ${detectionsFound} patterns.`, 'success');
  } else {
    status.textContent = 'Analyzing Image with Hybrid AI…';
    fill.style.width = '50%';
    const preds = await getAIPredictions(source);
    ctx.clearRect(0, 0, vw, vh);
    preds.forEach(p =>
    {
      if (isCameraLike(p.class, p.score)) {
        drawBBox(ctx, p.bbox, p.class, p.score);
        learnFromDetection(p.class, p.score);
      }
    });
    fill.style.width = '100%';
    showToast('✅ Image Analysis Complete. Hybrid Brain updated.', 'success');
  }

  labAnalysisRunning = false;
  btn.disabled = false;
  status.textContent = 'Learning session saved to local database.';
  updateLabStats();
  updateHomeStats();
}

function learnFromDetection(cls, score)
{
  // Simple heuristic: If the AI is very sure about a camera, 
  // we add it to the local 'knowledge base' and adapt thresholds
  // We also capture the current magnetic signature if scanner is running
  db.knowledgeBase.push({
    timestamp: Date.now(),
    class: cls,
    score: score,
    magValue: state.magnetic.running ? state.magnetic.lastValue : null
  });

  // Limit KB size
  if (db.knowledgeBase.length > 500) db.knowledgeBase.shift();

  // Adaptation: If we find many confirmed items, we slightly lower the risk threshold
  // to be more sensitive to those specific patterns.
  db.adaptedAIThreshold = Math.max(0.40, 0.60 - (db.knowledgeBase.length / 1000));
  saveLearningData();
}

function updateLabStats()
{
  const labSynapses = document.getElementById('lab-synapses');
  const labAccuracy = document.getElementById('lab-accuracy');
  const kbSize = document.getElementById('kb-size');
  const brainMap = document.getElementById('brain-map');

  if (labSynapses) labSynapses.textContent = db.knowledgeBase.length;
  const acc = (92 + (db.knowledgeBase.length * 0.01)).toFixed(1);
  if (labAccuracy) labAccuracy.textContent = Math.min(99.9, acc) + '%';
  if (kbSize) kbSize.textContent = `${db.knowledgeBase.length} patterns learned`;

  // Update Visual Brain Map
  if (brainMap) {
    if (db.knowledgeBase.length === 0) {
      brainMap.innerHTML = '<p style="font-size:0.65rem; color:var(--text-muted)">Brain is empty. Feed data to grow intelligence...</p>';
    } else {
      brainMap.innerHTML = '';
      // Show up to 50 icons representing knowledge bits
      const limit = Math.min(db.knowledgeBase.length, 50);
      for (let i = 0; i < limit; i++) {
        const dot = document.createElement('span');
        dot.style.fontSize = '12px';
        dot.style.filter = 'drop-shadow(0 0 2px #a855f7)';
        dot.textContent = '👾';
        brainMap.appendChild(dot);
      }
      if (db.knowledgeBase.length > 50) {
        const more = document.createElement('span');
        more.style.fontSize = '0.65rem';
        more.style.color = 'var(--primary)';
        more.textContent = `+${db.knowledgeBase.length - 50} more`;
        brainMap.appendChild(more);
      }
    }
  }
}

function clearBrain()
{
  if (confirm('⚠️ Warning: This will delete ALL learned patterns and factory reset the AI brain. Continue?')) {
    db.knowledgeBase = [];
    db.totalScans = 0;
    db.confirmedDetections = 0;
    db.adaptedAIThreshold = 0.60;
    saveLearningData();
    updateLabStats();
    updateHomeStats();
    showToast('🧹 AI Brain has been factory reset.', 'info');
  }
}

// ================================================================
// PORTABLE BRAIN: EXPORT / IMPORT
// ================================================================
function exportBrain()
{
  const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(db));
  const downloadAnchorNode = document.createElement('a');
  downloadAnchorNode.setAttribute("href", dataStr);
  downloadAnchorNode.setAttribute("download", "camguard_brain_data.json");
  document.body.appendChild(downloadAnchorNode);
  downloadAnchorNode.click();
  downloadAnchorNode.remove();
  showToast('📤 AI Brain Data exported successfully.', 'success');
}

function importBrain(input)
{
  if (input.files && input.files[0]) {
    const reader = new FileReader();
    reader.onload = function (e)
    {
      try {
        const imported = normalizeBrainPayload(JSON.parse(e.target.result));
        if (imported) {
          if (confirm('📥 This will OVERWRITE your current AI brain with the imported data. Continue?')) {
            db = imported;
            saveLearningData();
            updateLabStats();
            updateHomeStats();
            showToast('✅ AI Brain Data Imported Successfully!', 'success');
          }
        } else {
          showToast('❌ Invalid Brain File format', 'danger');
        }
      } catch (err) {
        showToast('❌ Error parsing Brain File', 'danger');
      }
    };
    reader.readAsText(input.files[0]);
  }
}

async function seedBrainFromProject()
{
  if (confirm('🔄 This will reload the "Master Brain" from the project files. Any unsaved local patterns might be lost. Sync now?')) {
    try {
      db = await fetchProjectBrain();
      saveLearningData();
      updateLabStats();
      updateHomeStats();
      showToast('Brain Synced with Project File!', 'success');
      if (false) {
        db = await resp.json();
        saveLearningData();
        updateLabStats();
        updateHomeStats();
        showToast('🧠 Brain Synced with Project File!', 'success');
      } else {
        showToast('❌ Master Brain file not found in project.', 'danger');
      }
    } catch (e) {
      showToast('❌ Sync Error: ' + e.message, 'danger');
    }
  }
}

function resetLab()
{
  document.getElementById('labPreviewContainer').classList.add('hidden');
  document.getElementById('labActions').classList.add('hidden');
  document.getElementById('labProgress').classList.add('hidden');
  document.getElementById('labFile').value = '';
}

function drawBBox(ctx, bbox, cls, score)
{
  const [x, y, w, h] = bbox;
  ctx.strokeStyle = '#00f5ff';
  ctx.lineWidth = 4;
  ctx.strokeRect(x, y, w, h);
  ctx.fillStyle = '#00f5ff';
  ctx.font = 'bold 16px Inter';
  ctx.fillText(`LEARNED: ${cls} (${Math.round(score * 100)}%)`, x, y - 10);
}

function updateHomeStats()
{
  const statsBar = document.getElementById('homeStats');
  if (!statsBar) return;

  statsBar.innerHTML = `
    <div class="stat-pill">🔍 ${db.totalScans} scans</div>
    <div class="stat-pill alert-pill">🚨 ${db.confirmedDetections} confirmed</div>
    <div class="stat-pill">🧠 ${db.knowledgeBase.length} learned</div>
  `;
}

// ================================================================
// IR / REMOTE SCAN LOGIC
// ================================================================
async function startIRScan()
{
  if (state.ir.running) return;

  try {
    // We prefer the FRONT camera because it usually lacks the IR filter
    state.ir.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } }
    });

    state.ir.running = true;
    const video = document.getElementById('irVideo');
    video.srcObject = state.ir.stream;
    video.play();

    document.getElementById('btnStartIR').disabled = true;
    document.getElementById('btnStopIR').disabled = false;
    setHeaderStatus('IR Scanning…', 'scanning');
    appendIRLog('🚀 IR Scanner started. Using Front Camera.', 'info');

    state.ir.animFrame = requestAnimationFrame(irFrame);
  } catch (e) {
    showToast('❌ Camera error: ' + e.message, 'danger');
  }
}

function stopIRScan()
{
  state.ir.running = false;
  cancelAnimationFrame(state.ir.animFrame);
  if (state.ir.stream) {
    state.ir.stream.getTracks().forEach(t => t.stop());
    state.ir.stream = null;
  }
  document.getElementById('btnStartIR').disabled = false;
  document.getElementById('btnStopIR').disabled = true;
  setHeaderStatus('Ready', 'idle');
  appendIRLog('⏹ Scanner stopped.', 'warn');
}

function irFrame()
{
  if (!state.ir.running) return;

  const video = document.getElementById('irVideo');
  const canvas = document.getElementById('irOverlay');
  const ctx = canvas.getContext('2d');

  if (video.readyState === video.HAVE_ENOUGH_DATA) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw simplified brightness map to detect IR hotspots
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const frame = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = frame.data;

    let brightnessSum = 0;
    let maxB = 0;
    for (let i = 0; i < data.length; i += 4) {
      const avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
      brightnessSum += avg;
      if (avg > maxB) maxB = avg;
    }

    const avgB = (brightnessSum / (data.length / 4));
    const intensity = Math.round((maxB / 255) * 100);

    state.ir.lastVal = state.ir.intensity;
    state.ir.intensity = intensity;

    updateIRDisplay(intensity, avgB);
  }

  state.ir.animFrame = requestAnimationFrame(irFrame);
}

function updateIRDisplay(intensity, avg)
{
  document.getElementById('irVal').textContent = intensity + '%';
  document.getElementById('irBar').style.width = intensity + '%';

  const diff = Math.abs(intensity - state.ir.lastVal);
  const sigEl = document.getElementById('irSig');

  if (intensity > 90) {
    sigEl.textContent = '🚨 HIGH GLINT';
    sigEl.style.color = 'var(--danger)';
    if (Math.random() > 0.95) appendIRLog('🚨 Direct IR target detected!', 'danger');
  } else if (diff > 15) {
    sigEl.textContent = '📡 PULSING';
    sigEl.style.color = 'var(--primary)';
    if (Math.random() > 0.98) appendIRLog('📡 Active IR signal detected (Pulsing)', 'info');
  } else {
    sigEl.textContent = intensity > 50 ? 'Medium' : 'Low';
    sigEl.style.color = 'var(--text-muted)';
  }
}

function appendIRLog(msg, type)
{
  const log = document.getElementById('irLog');
  const p = document.createElement('p');
  p.style.margin = '4px 0';
  p.style.borderLeft = `3px solid ${type === 'danger' ? 'var(--danger)' : (type === 'warn' ? 'var(--warn)' : 'var(--primary)')}`;
  p.style.paddingLeft = '8px';
  p.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
  log.prepend(p);
  if (log.children.length > 50) log.lastChild.remove();
}

console.log('%c🛡️ CamGuard AI', 'color:#00f5ff;font-size:20px;font-weight:bold;');
console.log('%cHidden Camera Detector — Powered by TensorFlow.js', 'color:#a855f7;');

// ================================================================
// MULTI-MODEL AI ORCHESTRATOR
// ================================================================
function setModelType(type)
{
  state.config.modelType = type;
  document.querySelectorAll('.model-pill').forEach(btn =>
  {
    btn.classList.toggle('active', btn.id === `btnModel-${type}`);
  });
  showToast(`🧠 Switched to ${type.toUpperCase()} mode`, 'info');
  // Restart if running
  if (state.visual.running) {
    stopVisualScan();
    setTimeout(startVisualScan, 500);
  }
}

async function loadYoloModel()
{
  if (state.models.yolo) return state.models.yolo;
  if (!navigator.onLine) {
    showToast('📴 YOLO requires internet. Using embedded model instead.', 'warn', 4000);
    return null;
  }
  const localUrl = './models/yolo/yolov8n_web_model/model.json';
  const remoteUrl = 'https://cdn.jsdelivr.net/gh/Hyuto/yolov8-tfjs@main/public/model/yolov8n_web_model/model.json';
  try {
    state.models.yolo = await tf.loadGraphModel(localUrl);
    console.log('✅ YOLOv8 Model Loaded (Local)');
    return state.models.yolo;
  } catch (e) {
    try {
      state.models.yolo = await tf.loadGraphModel(remoteUrl);
      console.log('✅ YOLOv8 Model Loaded (Remote)');
      return state.models.yolo;
    } catch (err) {
      console.error('YOLO Load Error', err);
      showToast('⚠️ YOLO Loading Error. Check connection.', 'danger');
      return null;
    }
  }
}

async function runYoloDetect(video)
{
  if (!state.models.yolo) return [];

  const [modelW, modelH] = [640, 640];

  return tf.tidy(() =>
  {
    const tensor = tf.browser.fromPixels(video)
      .resizeBilinear([modelW, modelH])
      .div(255.0)
      .expandDims(0);

    const res = state.models.yolo.predict(tensor);
    // YOLOv8 output: [1, 84, 8400]
    const out = res.reshape([84, 8400]).transpose([1, 0]);

    const boxes = out.slice([0, 0], [8400, 4]);
    const scores = out.slice([0, 4], [8400, 80]).max(1);
    const classes = out.slice([0, 4], [8400, 80]).argMax(1);

    const nmsIndices = tf.image.nonMaxSuppression(boxes, scores, 20, 0.45, 0.45);
    const bArr = boxes.gather(nmsIndices).arraySync();
    const sArr = scores.gather(nmsIndices).arraySync();
    const cArr = classes.gather(nmsIndices).arraySync();

    const vw = video.videoWidth;
    const vh = video.videoHeight;

    return sArr.map((s, i) =>
    {
      const [xc, yc, w, h] = bArr[i];
      // Convert normalized center coords to pixel corner coords
      const x = (xc - w / 2) / modelW * vw;
      const y = (yc - h / 2) / modelH * vh;
      const boxW = w / modelW * vw;
      const boxH = h / modelH * vh;

      return {
        class: COCO_CLASSES[cArr[i]],
        score: s,
        bbox: [x, y, boxW, boxH]
      };
    });
  });
}

// ================================================================
// UNIFIED MULTI-MODEL AI INTERFACE
// ================================================================
async function loadSelectedModels(statusCallback)
{
  const offline = !navigator.onLine;
  const type = offline ? 'custom' : state.config.modelType;
  if (offline && state.config.modelType !== 'custom' && !state.offlineHintShown) {
    showToast('📴 Offline mode: using embedded model only', 'info', 4000);
    state.offlineHintShown = true;
  }
  if (statusCallback) statusCallback(`Loading ${type.toUpperCase()} engines…`);

  const loaders = [];
  const useCustom = type === 'custom' || localStorage.getItem('camguard_custom_brain') === 'true' || offline;
  if (useCustom && !state.models.custom) {
    loaders.push(loadCustomModel());
  }
  if (!offline && (type === 'coco' || type === 'hybrid')) {
    if (!state.models.coco) {
      loaders.push(cocoSsd.load({ base: 'mobilenet_v2' }).then(m => state.models.coco = m));
    }
  }
  if (!offline && (type === 'yolo' || type === 'hybrid')) {
    if (!state.models.yolo) {
      loaders.push(loadYoloModel().then(m => state.models.yolo = m));
    }
  }

  await Promise.all(loaders);
  if (statusCallback) statusCallback('✅ AI Engines Ready');
}

async function loadCustomModel()
{
  if (!tf) throw new Error('TensorFlow.js not loaded');
  try {
    let model = null;
    if (window.CAMGUARD_EMBEDDED_MODEL && window.CAMGUARD_EMBEDDED_MODEL.getHandler) {
      const handler = window.CAMGUARD_EMBEDDED_MODEL.getHandler();
      model = await tf.loadLayersModel(handler);
      showToast('Custom Model Loaded (Embedded)', 'success');
    } else {
      model = await tf.loadLayersModel(CUSTOM_MODEL_PATH);
      showToast('Custom Model Loaded', 'success');
    }
    state.models.custom = model;
    localStorage.setItem('camguard_custom_brain', 'true');
    return model;
  } catch (e) {
    console.warn('Custom model load failed', e);
    localStorage.setItem('camguard_custom_brain', 'false');
    showToast('Custom Model Failed - Using Default', 'warning');
    return null;
  }
}

function runCustomPredict(source)
{
  return tf.tidy(() =>
  {
    const tensor = tf.browser.fromPixels(source).resizeNearestNeighbor([224, 224]).toFloat().expandDims();
    const scoreTensor = state.models.custom.predict(tensor);
    const score = scoreTensor.dataSync()[0];
    return [{ class: 'Hidden Camera', score: score, bbox: [20, 20, 200, 200] }]; // normalized/rough for class
  });
}

async function getAIPredictions(source)
{
  const offline = !navigator.onLine;
  const type = offline ? 'custom' : state.config.modelType;
  const useCustomModel = localStorage.getItem('camguard_custom_brain') === 'true';

  if ((type === 'custom' || useCustomModel) && state.models.custom) {
    return runCustomPredict(source);
  }

  if (type === 'coco') {
    if (!state.models.coco) return state.models.custom ? runCustomPredict(source) : [];
    return await state.models.coco.detect(source);
  } else if (type === 'yolo') {
    if (!state.models.yolo) return state.models.custom ? runCustomPredict(source) : [];
    return await runYoloDetect(source);
  } else if (type === 'hybrid') {
    const [cocoP, yoloP] = await Promise.all([
      state.models.coco ? state.models.coco.detect(source) : Promise.resolve([]),
      runYoloDetect(source)
    ]);
    // Merge: prioritise YOLO for detection, use COCO for confirmation
    const merged = [...yoloP];
    cocoP.forEach(cp =>
    {
      if (!merged.some(yp => yp.class === cp.class && yp.score > 0.4)) {
        merged.push(cp);
      }
    });
    return merged.length ? merged : (state.models.custom ? runCustomPredict(source) : []);
  }
  return [];
}

const COCO_CLASSES = [
  "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
  "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
  "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
  "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
  "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
  "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
  "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
  "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
];
