/**
 * VLA Embodied Memory & MuJoCo 3D Digital Twin Dashboard Client
 * -----------------------------------------------------------
 * Features:
 * - Three.js WebGL 3D Scene with MuJoCo Checkerboard Floor
 * - 7-DoF Franka Emika Panda Robot Arm Kinematic Mesh Animation
 * - Real-time SSE / EventSource Stream for MuJoCo Off-screen Camera View & Telemetry
 */

// Global State & Elements
let scene, camera, renderer, controls;
let robotArm = {};
let targetMesh, tableMesh, wallMesh;
let lastFrameTime = performance.now();
let frameCount = 0;
let currentCameraName = "overhead_cam";

// DOM Elements
const streamFrameEl = document.getElementById('stream-frame');
const overlayPromptEl = document.getElementById('overlay-prompt-text');
const fpsValEl = document.getElementById('fps-val');
const occlusionValEl = document.getElementById('occlusion-val');
const vlaModelNameEl = document.getElementById('vla-model-name');
const taskStatusValEl = document.getElementById('task-status-val');
const taskStatusNameEl = document.getElementById('task-status-name');
const taskInstructionBadgeEl = document.getElementById('task-instruction-badge');
const taskResultBannerEl = document.getElementById('task-result-banner');
const taskResultTextEl = document.getElementById('task-result-text');

// Telemetry DOM Elements
const tcpXyzEl = document.getElementById('tcp-xyz-val');
const targetXyzEl = document.getElementById('target-xyz-val');
const jointEls = Array.from({ length: 7 }, (_, i) => document.getElementById(`j${i + 1}-val`));
const grpValEl = document.getElementById('grp-val');

// Chronicler Log DOM
const logListEl = document.getElementById('chronicler-log-list');

// Control Buttons & Inputs
const commandInputEl = document.getElementById('human-command-input');
const btnSendCommand = document.getElementById('btn-send-command');
const btnResetSim = document.getElementById('btn-reset-sim');
const btnMoveTarget = document.getElementById('btn-reposition-target');
const btnResetView = document.getElementById('btn-reset-view');
const camSelectEl = document.getElementById('cam-select');


// ==========================================
// 1. THREE.JS 3D DIGITAL TWIN INITIALIZATION
// ==========================================

function initThreeScene() {
  const container = document.getElementById('three-canvas-container');
  const width = container.clientWidth;
  const height = container.clientHeight;

  // Scene & Background
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x121721);
  scene.fog = new THREE.FogExp2(0x121721, 0.15);

  // Camera
  camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
  camera.position.set(1.5, 1.2, 1.2);

  // Renderer with Shadows & ACES Tone Mapping
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;
  container.appendChild(renderer.domElement);

  // Orbit Controls
  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.target.set(0.4, 0.0, 0.4);
  controls.update();

  // Lighting (Matching MuJoCo XML headlight & directional light)
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
  scene.add(ambientLight);

  const hemiLight = new THREE.HemisphereLight(0xffffff, 0x333333, 0.6);
  hemiLight.position.set(0, 5, 0);
  scene.add(hemiLight);

  const dirLight = new THREE.DirectionalLight(0xffffff, 1.0);
  dirLight.position.set(2, 4, 3);
  dirLight.castShadow = true;
  dirLight.shadow.mapSize.width = 2048;
  dirLight.shadow.mapSize.height = 2048;
  dirLight.shadow.camera.near = 0.5;
  dirLight.shadow.camera.far = 10;
  dirLight.shadow.camera.left = -1.5;
  dirLight.shadow.camera.right = 1.5;
  dirLight.shadow.camera.top = 1.5;
  dirLight.shadow.camera.bottom = -1.5;
  scene.add(dirLight);

  // Create MuJoCo Checkerboard Floor & Environment
  createMuJoCoFloor();
  createEnvironmentObjects();
  createFrankaPandaKinematicMesh();

  // Window Resize Handler
  window.addEventListener('resize', onWindowResize);

  // Start Render Loop
  animateThree();
}

/**
 * Creates a procedural Canvas Texture matching MuJoCo's signature checkerboard plane.
 */
function createMuJoCoFloor() {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext('2d');

  const numSquares = 8;
  const size = 512 / numSquares;

  for (let i = 0; i < numSquares; i++) {
    for (let j = 0; j < numSquares; j++) {
      // Colors matching MuJoCo checkerboard plane (rgb1: 0.2, rgb2: 0.3)
      ctx.fillStyle = (i + j) % 2 === 0 ? '#33383e' : '#474f5a';
      ctx.fillRect(i * size, j * size, size, size);
      
      // Subtle grid border
      ctx.strokeStyle = '#252a30';
      ctx.lineWidth = 2;
      ctx.strokeRect(i * size, j * size, size, size);
    }
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(5, 5);

  const floorGeo = new THREE.PlaneGeometry(6, 6);
  const floorMat = new THREE.MeshStandardMaterial({
    map: texture,
    roughness: 0.7,
    metalness: 0.1
  });

  const floor = new THREE.Mesh(floorGeo, floorMat);
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = 0;
  floor.receiveShadow = true;
  scene.add(floor);
}

/**
 * Creates realistic Campbell's Tomato Soup Can label texture matching YCB dataset.
 */
function createSoupCanTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 256;
  const ctx = canvas.getContext('2d');

  // Top half: Campbell Red
  ctx.fillStyle = '#b71234';
  ctx.fillRect(0, 0, 512, 128);

  // Bottom half: Creamy White
  ctx.fillStyle = '#f8f8f6';
  ctx.fillRect(0, 128, 512, 128);

  // Top rim & bottom rim metallic accents
  ctx.fillStyle = '#a0a0a0';
  ctx.fillRect(0, 0, 512, 8);
  ctx.fillRect(0, 248, 512, 8);

  // Campbell's cursive script in top half
  ctx.fillStyle = '#ffffff';
  ctx.font = 'bold 36px "Brush Script MT", cursive, serif';
  ctx.textAlign = 'center';
  ctx.fillText("Campbell's", 256, 75);

  // Gold Medallion in center
  ctx.beginPath();
  ctx.arc(256, 128, 20, 0, Math.PI * 2);
  ctx.fillStyle = '#d4af37';
  ctx.fill();
  ctx.lineWidth = 2;
  ctx.strokeStyle = '#b8972e';
  ctx.stroke();

  // Bottom text: CONDENSED TOMATO SOUP
  ctx.fillStyle = '#b71234';
  ctx.font = 'bold 18px "Arial Black", sans-serif';
  ctx.fillText("TOMATO", 256, 185);

  ctx.fillStyle = '#333333';
  ctx.font = 'bold 12px Arial, sans-serif';
  ctx.fillText("SOUP", 256, 210);

  const texture = new THREE.CanvasTexture(canvas);
  return texture;
}

/**
 * Creates Table surface and Target object matching franka_table_scene.xml.
 */
function createEnvironmentObjects() {
  // Table Top (pos: [0.5, 0, 0.35], size: [0.45, 0.55, 0.02])
  const tableGeo = new THREE.BoxGeometry(0.9, 1.1, 0.04);
  const tableMat = new THREE.MeshStandardMaterial({
    color: 0xd1b38e, // Table wood color
    roughness: 0.5,
    metalness: 0.1
  });
  tableMesh = new THREE.Mesh(tableGeo, tableMat);
  tableMesh.position.set(0.5, 0.35, 0); // MuJoCo Z is UP, Three.js Y is UP (we map Z -> Y, Y -> -Z)
  // Let's use MuJoCo coordinates transform: (X_mj, Z_mj, -Y_mj)
  tableMesh.position.set(0.5, 0.35, 0.0);
  tableMesh.castShadow = true;
  tableMesh.receiveShadow = true;
  scene.add(tableMesh);

  // Table Legs
  const legMat = new THREE.MeshStandardMaterial({ color: 0x333333, metalness: 0.8 });
  const legGeo = new THREE.CylinderGeometry(0.03, 0.03, 0.34);
  const legPositions = [
    [0.9, 0.17, 0.5],
    [0.1, 0.17, 0.5],
    [0.9, 0.17, -0.5],
    [0.1, 0.17, -0.5]
  ];
  legPositions.forEach(pos => {
    const leg = new THREE.Mesh(legGeo, legMat);
    leg.position.set(pos[0], pos[1], pos[2]);
    leg.castShadow = true;
    scene.add(leg);
  });

  // Target Object: YCB Campbell's Tomato Soup Can
  const canTexture = createSoupCanTexture();
  const targetGeo = new THREE.CylinderGeometry(0.034, 0.034, 0.10, 32);
  const targetMat = new THREE.MeshStandardMaterial({
    map: canTexture,
    roughness: 0.35,
    metalness: 0.25
  });
  targetMesh = new THREE.Mesh(targetGeo, targetMat);
  targetMesh.position.set(0.55, 0.40, 0.15);
  targetMesh.castShadow = true;
  targetMesh.receiveShadow = true;
  scene.add(targetMesh);

  // Occlusion Wall (Transparent Gray Box)
  const wallGeo = new THREE.BoxGeometry(0.04, 0.25, 0.24);
  const wallMat = new THREE.MeshStandardMaterial({
    color: 0x666677,
    transparent: true,
    opacity: 0.6,
    roughness: 0.3
  });
  wallMesh = new THREE.Mesh(wallGeo, wallMat);
  wallMesh.position.set(0.42, 0.495, 0.0);
  wallMesh.castShadow = true;
  scene.add(wallMesh);
}

/**
 * Constructs the 7-DoF Franka Emika Panda robot kinematic mesh hierarchy.
 */
function createFrankaPandaKinematicMesh() {
  const robotGroup = new THREE.Group();
  robotGroup.position.set(0.0, 0.37, 0.0); // Base at table edge

  // Metallic & Plastic Materials matching Franka Panda white & dark grey
  const whiteMat = new THREE.MeshStandardMaterial({ color: 0xf5f5f5, roughness: 0.3, metalness: 0.2 });
  const darkMat = new THREE.MeshStandardMaterial({ color: 0x2b2b2b, roughness: 0.4, metalness: 0.6 });
  const gripperMat = new THREE.MeshStandardMaterial({ color: 0x3a3a3a, roughness: 0.3, metalness: 0.5 });

  // Base Link 0
  const link0 = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.07, 0.08), darkMat);
  link0.position.y = 0.04;
  link0.castShadow = true;
  robotGroup.add(link0);

  // Joint 1 Group (Yaw)
  const j1Group = new THREE.Group();
  j1Group.position.y = 0.08;
  const link1 = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.05, 0.16), whiteMat);
  link1.position.y = 0.08;
  link1.castShadow = true;
  j1Group.add(link1);

  // Joint 2 Group (Pitch)
  const j2Group = new THREE.Group();
  j2Group.position.y = 0.16;
  const link2 = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.045, 0.20), whiteMat);
  link2.position.y = 0.10;
  link2.castShadow = true;
  j2Group.add(link2);

  // Joint 3 Group (Roll)
  const j3Group = new THREE.Group();
  j3Group.position.y = 0.20;
  const link3 = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 0.18), whiteMat);
  link3.position.set(0.04, 0.09, 0.0);
  link3.castShadow = true;
  j3Group.add(link3);

  // Joint 4 Group (Elbow Pitch)
  const j4Group = new THREE.Group();
  j4Group.position.set(0.04, 0.18, 0.0);
  const link4 = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 0.20), whiteMat);
  link4.position.y = 0.10;
  link4.castShadow = true;
  j4Group.add(link4);

  // Joint 5 Group (Forearm Roll)
  const j5Group = new THREE.Group();
  j5Group.position.y = 0.20;
  const link5 = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 0.15), whiteMat);
  link5.position.y = 0.075;
  link5.castShadow = true;
  j5Group.add(link5);

  // Joint 6 Group (Wrist Pitch)
  const j6Group = new THREE.Group();
  j6Group.position.y = 0.15;
  const link6 = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 0.08), darkMat);
  link6.position.y = 0.04;
  link6.castShadow = true;
  j6Group.add(link6);

  // Joint 7 Group (Wrist Roll)
  const j7Group = new THREE.Group();
  j7Group.position.y = 0.08;
  const link7 = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, 0.06), whiteMat);
  link7.position.y = 0.03;
  link7.castShadow = true;
  j7Group.add(link7);

  // Gripper Hand
  const handGroup = new THREE.Group();
  handGroup.position.y = 0.06;
  const handMesh = new THREE.Mesh(new THREE.BoxGeometry(0.07, 0.04, 0.04), gripperMat);
  handMesh.castShadow = true;
  handGroup.add(handMesh);

  // Left Finger
  const leftFinger = new THREE.Mesh(new THREE.BoxGeometry(0.016, 0.05, 0.01), darkMat);
  leftFinger.position.set(0, 0.04, -0.02);
  leftFinger.castShadow = true;
  handGroup.add(leftFinger);

  // Right Finger
  const rightFinger = new THREE.Mesh(new THREE.BoxGeometry(0.016, 0.05, 0.01), darkMat);
  rightFinger.position.set(0, 0.04, 0.02);
  rightFinger.castShadow = true;
  handGroup.add(rightFinger);

  // Hierarchy Assembly
  link0.add(j1Group);
  j1Group.add(j2Group);
  j2Group.add(j3Group);
  j3Group.add(j4Group);
  j4Group.add(j5Group);
  j5Group.add(j6Group);
  j6Group.add(j7Group);
  j7Group.add(handGroup);

  scene.add(robotGroup);

  // Store references for joint angle updates
  robotArm = {
    group: robotGroup,
    j1: j1Group,
    j2: j2Group,
    j3: j3Group,
    j4: j4Group,
    j5: j5Group,
    j6: j6Group,
    j7: j7Group,
    leftFinger: leftFinger,
    rightFinger: rightFinger
  };
}

/**
 * Updates 3D Franka Panda arm joint angles and gripper spacing matching MuJoCo qpos.
 */
function updateRobotKinematics(qpos, fingerPos) {
  if (!robotArm.j1 || !qpos || qpos.length < 7) return;

  // Apply joint rotations (mapping MuJoCo joint axes to Three.js axes)
  robotArm.j1.rotation.y = qpos[0];
  robotArm.j2.rotation.z = qpos[1];
  robotArm.j3.rotation.y = qpos[2];
  robotArm.j4.rotation.z = qpos[3];
  robotArm.j5.rotation.y = qpos[4];
  robotArm.j6.rotation.z = qpos[5];
  robotArm.j7.rotation.y = qpos[6];

  // Update parallel gripper finger slide offsets
  const fingerOffset = fingerPos || 0.04;
  if (robotArm.leftFinger && robotArm.rightFinger) {
    robotArm.leftFinger.position.z = -fingerOffset;
    robotArm.rightFinger.position.z = fingerOffset;
  }
}

function animateThree() {
  requestAnimationFrame(animateThree);
  controls.update();
  renderer.render(scene, camera);
}

function onWindowResize() {
  const container = document.getElementById('three-canvas-container');
  if (!container) return;
  const width = container.clientWidth;
  const height = container.clientHeight;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height);
}


// ==========================================
// 2. REAL-TIME STREAMING & TELEMETRY CLIENT
// ==========================================

function initStreamConnection() {
  const streamUrl = `/stream?camera=${currentCameraName}`;
  const evtSource = new EventSource(streamUrl);

  evtSource.onmessage = function (event) {
    try {
      const data = JSON.parse(event.data);

      // 1. Update MuJoCo Offscreen Perception Frame
      if (data.frame_b64) {
        streamFrameEl.src = data.frame_b64;
      }

      // 2. Update Overlay Dynamic Prompt Banner
      if (data.dynamic_prompt) {
        overlayPromptEl.innerText = data.dynamic_prompt;
      }

      // 3. Update VLA Model Name & Metrics Badges
      if (data.vla_model) {
        vlaModelNameEl.innerText = data.vla_model;
      }

      if (data.task_name && taskStatusNameEl) {
        taskStatusNameEl.innerText = data.task_name.replace(/_/g, ' ').toUpperCase();
      }

      if (data.instruction && taskInstructionBadgeEl) {
        taskInstructionBadgeEl.innerText = `Instruction: "${data.instruction}"`;
      }

      // 4. Update Task Status & Banner (SUCCESS / FAILED / RUNNING)
      const currentStatus = data.status || (data.telemetry && data.telemetry.status) || "RUNNING";
      if (taskStatusValEl) {
        taskStatusValEl.innerText = currentStatus;
        if (currentStatus === "SUCCESS") {
          taskStatusValEl.className = "text-green font-bold";
        } else if (currentStatus === "FAILED") {
          taskStatusValEl.className = "text-red font-bold";
        } else {
          taskStatusValEl.className = "text-blue";
        }
      }

      if (taskResultBannerEl) {
        if (currentStatus === "SUCCESS") {
          taskResultBannerEl.className = "task-result-banner banner-success";
          if (taskResultTextEl) taskResultTextEl.innerText = "TASK SUCCESSFUL! (Goal Reached)";
        } else if (currentStatus === "FAILED") {
          taskResultBannerEl.className = "task-result-banner banner-failed";
          if (taskResultTextEl) taskResultTextEl.innerText = "TASK FAILED (Max Steps Reached)";
        } else {
          taskResultBannerEl.className = "task-result-banner hidden";
        }
      }

      if (data.telemetry) {
        const tel = data.telemetry;

        // Update 3D Digital Twin Robot Joints
        updateRobotKinematics(tel.qpos, tel.finger);

        // Update Joint Telemetry Text
        if (tel.qpos && tel.qpos.length >= 7) {
          tel.qpos.forEach((q, idx) => {
            if (jointEls[idx]) {
              const deg = (q * (180 / Math.PI)).toFixed(1);
              jointEls[idx].innerText = `${deg}`;
            }
          });
        }

        // Gripper state
        if (grpValEl) {
          grpValEl.innerText = tel.finger < 0.015 ? "Closed" : "Open";
          grpValEl.className = tel.finger < 0.015 ? "text-red" : "text-green";
        }

        // Update TCP and Target Positions
        if (tel.tcp_pos) {
          tcpXyzEl.innerText = `[${tel.tcp_pos[0].toFixed(2)}, ${tel.tcp_pos[1].toFixed(2)}, ${tel.tcp_pos[2].toFixed(2)}] m`;
        }

        if (tel.target_pos) {
          targetXyzEl.innerText = `[${tel.target_pos[0].toFixed(2)}, ${tel.target_pos[1].toFixed(2)}, ${tel.target_pos[2].toFixed(2)}] m`;
          // Update 3D target object position in Three.js (map MuJoCo [X, Y, Z] -> Three.js [X, Z_up->Y, Y_lat->Z])
          if (targetMesh) {
            targetMesh.position.set(tel.target_pos[0], tel.target_pos[2], tel.target_pos[1]);
          }
        }

        // Update Occlusion Badge
        if (tel.metrics) {
          const isOccluded = tel.metrics.is_occluded;
          occlusionValEl.innerText = isOccluded ? "OCCLUDE" : "Visible";
          occlusionValEl.className = isOccluded ? "text-red" : "text-green";
        }
      }

      // Update Chronicler Memory Log
      if (data.chronicler_log) {
        appendChroniclerLog(data.chronicler_log);
      }

      // FPS calculation
      frameCount++;
      const now = performance.now();
      if (now - lastFrameTime >= 1000) {
        const fps = Math.round((frameCount * 1000) / (now - lastFrameTime));
        fpsValEl.innerText = `${fps}`;
        frameCount = 0;
        lastFrameTime = now;
      }

    } catch (e) {
      console.error("[Stream] Error parsing JSON event:", e);
    }
  };

  evtSource.onerror = function (err) {
    console.warn("[Stream] SSE Connection lost, retrying...", err);
  };
}

function appendChroniclerLog(logMsg) {
  if (!logMsg) return;
  
  // Prevent duplicate entries
  const lastLog = logListEl.firstElementChild;
  if (lastLog && lastLog.innerText.includes(logMsg)) return;

  const nowStr = new Date().toLocaleTimeString();
  const logItem = document.createElement('div');
  logItem.className = 'log-item';
  logItem.innerHTML = `<span class="log-time">[${nowStr}]</span><span class="log-prompt">${escapeHtml(logMsg)}</span>`;

  logListEl.insertBefore(logItem, logListEl.firstChild);

  // Limit log feed length to 30 items
  while (logListEl.children.length > 30) {
    logListEl.removeChild(logListEl.lastChild);
  }
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.innerText = text;
  return div.innerHTML;
}


// ==========================================
// 3. INTERACTIVE CONTROLS & API CALLS
// ==========================================

function setupEventListeners() {
  // Send Command Button
  btnSendCommand.addEventListener('click', sendInstructionCommand);
  commandInputEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') sendInstructionCommand();
  });

  // Reposition Target
  btnMoveTarget.addEventListener('click', () => {
    fetch('/api/move_target', { method: 'POST' })
      .then(res => res.json())
      .then(data => appendChroniclerLog(`[System]: Repositioned target to [${data.target_pos.map(n => n.toFixed(2)).join(', ')}]`))
      .catch(err => console.error(err));
  });

  // Reset Physics
  btnResetSim.addEventListener('click', () => {
    fetch('/api/reset', { method: 'POST' })
      .then(res => res.json())
      .then(data => appendChroniclerLog("[System]: Physics simulation reset to home keyframe."))
      .catch(err => console.error(err));
  });

  // Reset View Angle
  btnResetView.addEventListener('click', () => {
    camera.position.set(1.5, 1.2, 1.2);
    controls.target.set(0.4, 0.0, 0.4);
    controls.update();
  });

  // Camera Selection Dropdown
  camSelectEl.addEventListener('change', (e) => {
    currentCameraName = e.target.value;
    fetch(`/api/set_camera?name=${currentCameraName}`, { method: 'POST' })
      .then(res => res.json())
      .then(data => {
        appendChroniclerLog(`[Camera]: Switched viewpoint to ${data.active_camera}`);
      })
      .catch(err => console.error(err));
  });
}

function sendInstructionCommand() {
  const text = commandInputEl.value.trim();
  if (!text) return;

  fetch('/api/command', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ command: text })
  })
    .then(res => res.json())
    .then(data => {
      appendChroniclerLog(`[User Command]: "${text}"`);
    })
    .catch(err => console.error(err));
}


// Initialize Application
window.addEventListener('DOMContentLoaded', () => {
  initThreeScene();
  setupEventListeners();
  initStreamConnection();
});
