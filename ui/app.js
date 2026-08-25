/**
 * VisionBoard Studio — Frontend Application Engine
 * Exact match for Concept 1: Neon Glassmorphic AI Studio
 */

const state = {
  activeModel: 'yolov8x',
  activeDetectionIndex: 0,
  activeScene: 'city_speed',
  isScanning: false,
  customImage: null,
};

// Scene Data with Accurate Realistic Signboard Placements
const SCENES = {
  city_speed: {
    title: 'Urban Boulevard & Traffic Signs',
    detections: [
      {
        id: 1,
        className: 'SPEED_LIMIT',
        conf: 88,
        ocrText: '40 MPH (Speed Limit)',
        ocrRaw: 'SPEED 40',
        sideTag: 'SPEED_LIMIT ( 40 MPH )',
        bottomTag: '88% | 40 MPH',
        box: { top: 22, left: 57, width: 9.5, height: 26 }, // percentages
        category: 'Speed Limit',
        accuracyPct: 96.5
      },
      {
        id: 2,
        className: 'PED_CROSSING',
        conf: 92,
        ocrText: 'PEDESTRIAN CROSSING',
        ocrRaw: 'PED XING',
        sideTag: 'PED_CROSSING · TURN_LEFT (0)',
        bottomTag: '92% | PED XING',
        box: { top: 43, left: 24, width: 7.2, height: 16 },
        category: 'Warning',
        accuracyPct: 94.2
      },
      {
        id: 3,
        className: 'TURN_LEFT',
        conf: 84,
        ocrText: 'TURN LEFT WAY',
        ocrRaw: 'TURN LEFT',
        sideTag: 'TURN_LEFT WAYEN 84%',
        bottomTag: '84% | LEFT',
        box: { top: 52, left: 41, width: 6.8, height: 16 },
        category: 'Information',
        accuracyPct: 91.8
      }
    ]
  },
  highway_exit: {
    title: 'Interstate Highway Directional Signs',
    detections: [
      {
        id: 1,
        className: 'SPEED_LIMIT',
        conf: 95,
        ocrText: '65 MPH (Speed Limit)',
        ocrRaw: 'SPEED 65',
        sideTag: 'SPEED_LIMIT ( 65 MPH )',
        bottomTag: '95% | 65 MPH',
        box: { top: 20, left: 20, width: 12, height: 28 },
        category: 'Speed Limit',
        accuracyPct: 98.4
      },
      {
        id: 2,
        className: 'GUIDE_SIGN',
        conf: 91,
        ocrText: 'EXIT 42A METRO AIRPORT',
        ocrRaw: 'EXIT 42A',
        sideTag: 'EXIT_GUIDE ( AIRPORT )',
        bottomTag: '91% | AIRPORT',
        box: { top: 18, left: 55, width: 22, height: 24 },
        category: 'Information',
        accuracyPct: 97.1
      }
    ]
  }
};

// ==================== INITIALIZATION ====================
document.addEventListener('DOMContentLoaded', () => {
  renderSceneCanvas();
  renderBoundingBoxes();
  startLiveClock();
  setupInteractivity();
});

// ==================== CANVAS RENDERING ====================
function renderSceneCanvas() {
  const canvas = document.createElement('canvas');
  canvas.width = 1200;
  canvas.height = 700;
  const ctx = canvas.getContext('2d');

  if (state.activeScene === 'city_speed') {
    // Photorealistic urban street render
    const sky = ctx.createLinearGradient(0, 0, 0, 350);
    sky.addColorStop(0, '#1e293b');
    sky.addColorStop(0.6, '#334155');
    sky.addColorStop(1, '#64748b');
    ctx.fillStyle = sky;
    ctx.fillRect(0, 0, 1200, 350);

    // City Buildings
    ctx.fillStyle = '#111827';
    ctx.fillRect(40, 60, 240, 320);
    ctx.fillRect(320, 40, 260, 340);
    ctx.fillRect(620, 90, 220, 290);
    ctx.fillRect(880, 50, 280, 330);

    // Architectural window grids
    ctx.fillStyle = 'rgba(255, 237, 213, 0.15)';
    for (let r = 80; r < 340; r += 28) {
      for (let c = 60; c < 260; c += 24) ctx.fillRect(c, r, 12, 16);
      for (let c = 340; c < 560; c += 24) ctx.fillRect(c, r, 12, 16);
      for (let c = 900; c < 1140; c += 24) ctx.fillRect(c, r, 12, 16);
    }

    // Street Asphalt & Crosswalk
    ctx.fillStyle = '#1c1f26';
    ctx.fillRect(0, 350, 1200, 350);

    // Crosswalk Zebra Stripes
    ctx.fillStyle = 'rgba(255, 255, 255, 0.75)';
    for (let x = 320; x < 880; x += 55) {
      ctx.fillRect(x, 480, 35, 90);
    }

    // Road Lanes
    ctx.fillStyle = '#eab308';
    ctx.fillRect(200, 560, 80, 12);
    ctx.fillRect(920, 560, 80, 12);

    // Cars silhouettes
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(180, 410, 160, 70);
    ctx.fillRect(440, 400, 150, 65);
    ctx.fillRect(680, 415, 140, 60);

    // Pole for Speed Sign
    ctx.fillStyle = '#64748b';
    ctx.fillRect(738, 280, 16, 260);

    // Main Signboard 1: SPEED 40
    ctx.fillStyle = '#ffffff';
    ctx.strokeStyle = '#0f172a';
    ctx.lineWidth = 6;
    ctx.fillRect(690, 150, 114, 175);
    ctx.strokeRect(690, 150, 114, 175);

    ctx.fillStyle = '#0f172a';
    ctx.font = 'bold 22px Outfit, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('SPEED', 747, 195);
    ctx.font = 'bold 64px Outfit, sans-serif';
    ctx.fillText('40', 747, 275);

    // Signboard 2: Pedestrian Crossing (Yellow Diamond)
    ctx.save();
    ctx.translate(330, 350);
    ctx.rotate((45 * Math.PI) / 180);
    ctx.fillStyle = '#eab308';
    ctx.fillRect(-45, -45, 90, 90);
    ctx.restore();
    ctx.fillStyle = '#0f172a';
    ctx.font = 'bold 15px Outfit, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('PED XING', 330, 356);

    // Signboard 3: Turn Left
    ctx.fillStyle = '#15803d';
    ctx.fillRect(500, 365, 80, 90);
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 18px Outfit, sans-serif';
    ctx.fillText('← LEFT', 540, 415);
  }

  const imgEl = document.getElementById('main-image');
  imgEl.src = canvas.toDataURL('image/png');
}

// ==================== BOUNDING BOXES ====================
function renderBoundingBoxes() {
  const layer = document.getElementById('bounding-layer');
  layer.innerHTML = '';

  const scene = SCENES[state.activeScene];
  scene.detections.forEach((det, idx) => {
    const box = document.createElement('div');
    box.className = `neon-box ${idx === state.activeDetectionIndex ? 'active' : ''}`;
    box.id = `box-${idx}`;
    box.style.top = `${det.box.top}%`;
    box.style.left = `${det.box.left}%`;
    box.style.width = `${det.box.width}%`;
    box.style.height = `${det.box.height}%`;

    // Top Tag Badge
    const topTag = document.createElement('div');
    topTag.className = 'neon-tag-top';
    topTag.textContent = `${det.className} ${det.conf}%`;
    box.appendChild(topTag);

    // Side Tag Badge
    const sideTag = document.createElement('div');
    sideTag.className = 'neon-tag-side';
    sideTag.textContent = det.sideTag;
    box.appendChild(sideTag);

    // Bottom OCR Chip
    const botTag = document.createElement('div');
    botTag.className = 'neon-tag-bottom';
    botTag.textContent = det.bottomTag;
    box.appendChild(botTag);

    // Click/Hover Event
    box.addEventListener('click', () => selectDetection(idx));
    box.addEventListener('mouseenter', () => selectDetection(idx));

    layer.appendChild(box);
  });

  updateInspector(state.activeDetectionIndex);
}

function selectDetection(index) {
  state.activeDetectionIndex = index;

  // Highlight active box
  document.querySelectorAll('.neon-box').forEach((b, i) => {
    if (i === index) b.classList.add('active');
    else b.classList.remove('active');
  });

  updateInspector(index);
}

function updateInspector(index) {
  const scene = SCENES[state.activeScene];
  const det = scene.detections[index] || scene.detections[0];

  // OCR Box
  document.getElementById('ocr-extracted-text').textContent = det.ocrText;

  // Accuracy Dial & Stats
  document.getElementById('ocr-accuracy-pct').textContent = `${det.accuracyPct}%`;

  // Highlight stream list item
  document.querySelectorAll('.stream-item').forEach((item, i) => {
    if (i === index) item.classList.add('active');
    else item.classList.remove('active');
  });
}

// ==================== ACTIONS & CONTROLS ====================
function runDetection() {
  const btn = document.getElementById('btn-run');
  const laser = document.getElementById('scan-laser');
  const procEl = document.getElementById('top-processing');

  btn.style.opacity = '0.6';
  laser.classList.add('active');
  procEl.textContent = '82ms';

  showToast('Executing YOLOv8x inference + OCR...');

  setTimeout(() => {
    laser.classList.remove('active');
    btn.style.opacity = '1';
    procEl.textContent = '145ms';
    document.getElementById('tele-detections-count').textContent = '148';
    showToast('Detection complete (3 signboards identified)');
  }, 1200);
}

function changeModel() {
  state.activeModel = document.getElementById('model-select').value;
  document.getElementById('tele-model-name').textContent = state.activeModel.toUpperCase();
  const times = { yolov8x: '145ms', yolov8m: '64ms', yolov8s: '28ms', yolov8n: '12ms' };
  document.getElementById('top-processing').textContent = times[state.activeModel] || '145ms';
  showToast(`Switched model to ${state.activeModel.toUpperCase()}`);
}

function cycleScene() {
  state.activeScene = state.activeScene === 'city_speed' ? 'highway_exit' : 'city_speed';
  renderSceneCanvas();
  renderBoundingBoxes();
  showToast(`Switched scene: ${SCENES[state.activeScene].title}`);
}

function editOCRText() {
  const current = document.getElementById('ocr-extracted-text').textContent.trim();
  const updated = prompt('Edit extracted OCR text:', current);
  if (updated !== null && updated.trim() !== '') {
    document.getElementById('ocr-extracted-text').textContent = updated.trim();
    showToast('Updated OCR transcription');
  }
}

function triggerUpload() {
  document.getElementById('image-uploader').click();
}

function onCustomImage(event) {
  const file = event.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (e) => {
    document.getElementById('main-image').src = e.target.result;
    showToast(`Loaded user image: ${file.name}`);
  };
  reader.readAsDataURL(file);
}

function toggleFullscreen() {
  const box = document.getElementById('viewport-box');
  if (!document.fullscreenElement) {
    box.requestFullscreen().catch(err => {
      showToast(`Fullscreen error: ${err.message}`, 'warning');
    });
  } else {
    document.exitFullscreen();
  }
}

function switchNav(navKey, event) {
  if (event) event.preventDefault();
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  const target = document.getElementById(`nav-${navKey}`);
  if (target) target.classList.add('active');
  showToast(`Navigated to ${navKey.toUpperCase()}`);
}

function setupInteractivity() {
  document.querySelectorAll('.stream-item').forEach((item, idx) => {
    item.addEventListener('click', () => selectDetection(idx % SCENES[state.activeScene].detections.length));
  });
}

function startLiveClock() {
  function updateTime() {
    const now = new Date();
    const timeStr = now.toTimeString().split(' ')[0];
    const el = document.getElementById('live-timestamp');
    if (el) el.textContent = timeStr;
  }
  updateTime();
  setInterval(updateTime, 1000);
}

let toastTimer;
function showToast(msg) {
  const toast = document.getElementById('toast');
  const msgEl = document.getElementById('toast-msg');
  msgEl.textContent = msg;

  toast.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    toast.classList.remove('show');
  }, 2400);
}
