// ── Cortex — 3D Spiking Brain Visualization ────────────────────────
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js';
import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/controls/OrbitControls.js';

// ── Constants ───────────────────────────────────────────────────────
const BG_COLOR = 0x050510;
const TOP_BAR = 48;
const BOTTOM_BAR = 56;

const REGIONS = [
  { name: 'visual_cortex',     pos: [0, 0.5, -2.2],   color: 0x3b82f6, size: 0.45, label: 'Visual' },
  { name: 'auditory_cortex',   pos: [2, 0, -0.5],     color: 0xf59e0b, size: 0.4,  label: 'Auditory' },
  { name: 'association_cortex', pos: [0, 1, 0],        color: 0x10b981, size: 0.7,  label: 'Association' },
  { name: 'predictive_cortex', pos: [0, 1.5, -1],     color: 0x06b6d4, size: 0.4,  label: 'Predictive' },
  { name: 'hippocampus',       pos: [0, -0.3, 0],     color: 0xa855f7, size: 0.35, label: 'Hippocampus' },
  { name: 'prefrontal_cortex', pos: [0, 0.8, 2],      color: 0xef4444, size: 0.5,  label: 'Prefrontal' },
  { name: 'amygdala',          pos: [0, -0.8, 0.5],   color: 0xf59e0b, size: 0.3,  label: 'Amygdala' },
  { name: 'motor_cortex',      pos: [0, 2, 1],        color: 0x14b8a6, size: 0.4,  label: 'Motor' },
  { name: 'brainstem',         pos: [0, -2, 0],       color: 0x8b5cf6, size: 0.35, label: 'Brainstem' },
  { name: 'cerebellum',        pos: [0, -1.5, -1.5],  color: 0x67e8f9, size: 0.4,  label: 'Cerebellum' },
];

// ── Scene Setup ─────────────────────────────────────────────────────
const container = document.getElementById('brain-canvas');
const width = () => window.innerWidth;
const height = () => window.innerHeight - TOP_BAR - BOTTOM_BAR;

const scene = new THREE.Scene();
scene.background = new THREE.Color(BG_COLOR);

const camera = new THREE.PerspectiveCamera(50, width() / height(), 0.1, 100);
camera.position.set(0, 2, 8);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(width(), height());
container.appendChild(renderer.domElement);

// ── Controls ────────────────────────────────────────────────────────
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.3;
controls.minDistance = 3;
controls.maxDistance = 20;

// ── Lights ──────────────────────────────────────────────────────────
scene.add(new THREE.AmbientLight(0x404060, 0.6));
const pointLight = new THREE.PointLight(0x10b981, 1.2, 30);
pointLight.position.set(4, 6, 4);
scene.add(pointLight);

// ── Brain Shell ─────────────────────────────────────────────────────
const shellGeo = new THREE.SphereGeometry(3, 32, 24);
shellGeo.scale(1.2, 1, 1);
const shellMat = new THREE.MeshBasicMaterial({
  color: 0x10b981,
  wireframe: true,
  transparent: true,
  opacity: 0.08,
});
const brainShell = new THREE.Mesh(shellGeo, shellMat);
scene.add(brainShell);

// ── Brain Regions ───────────────────────────────────────────────────
const regionMeshes = {};

for (const r of REGIONS) {
  const geo = new THREE.SphereGeometry(r.size, 24, 18);
  const mat = new THREE.MeshPhongMaterial({
    color: r.color,
    emissive: r.color,
    emissiveIntensity: 0.15,
    transparent: true,
    opacity: 0.55,
    shininess: 60,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.set(...r.pos);
  mesh.userData = { name: r.name, baseEmissive: 0.15 };
  scene.add(mesh);
  regionMeshes[r.name] = mesh;
}

// ── Knowledge Nodes ─────────────────────────────────────────────────
const knowledgeGroup = new THREE.Group();
scene.add(knowledgeGroup);
const knowledgeNodes = {};  // label -> mesh
const edgeLines = [];

function topicHue(topic) {
  let h = 0;
  for (let i = 0; i < topic.length; i++) h = (h * 31 + topic.charCodeAt(i)) & 0xffff;
  return (h % 360) / 360;
}

function goldenAnglePosition(index, total, center, radius) {
  const golden = Math.PI * (3 - Math.sqrt(5));
  const y = 1 - (index / (total - 1 || 1)) * 2;
  const r = Math.sqrt(1 - y * y);
  const theta = golden * index;
  return new THREE.Vector3(
    center[0] + Math.cos(theta) * r * radius,
    center[1] + y * radius,
    center[2] + Math.sin(theta) * r * radius
  );
}

async function cortexLoadGraph() {
  try {
    const resp = await fetch('/api/brain/knowledge/graph');
    if (!resp.ok) return;
    const data = await resp.json();
    const concepts = data.concepts || [];
    const connections = data.connections || [];

    // Clear old
    while (knowledgeGroup.children.length) knowledgeGroup.remove(knowledgeGroup.children[0]);
    for (const l of edgeLines) scene.remove(l);
    edgeLines.length = 0;
    Object.keys(knowledgeNodes).forEach(k => delete knowledgeNodes[k]);

    const assocCenter = [0, 1, 0];
    const radius = 2.0;

    concepts.forEach((c, i) => {
      const hue = topicHue(c.topic || 'default');
      const col = new THREE.Color().setHSL(hue, 0.7, 0.6);
      const deg = c.degree || 1;
      const s = 0.04 + Math.min(deg, 20) * 0.008;
      const geo = new THREE.SphereGeometry(s, 8, 6);
      const mat = new THREE.MeshPhongMaterial({
        color: col,
        emissive: col,
        emissiveIntensity: 0.3,
        transparent: true,
        opacity: 0.8,
      });
      const mesh = new THREE.Mesh(geo, mat);
      const pos = goldenAnglePosition(i, concepts.length, assocCenter, radius);
      mesh.position.copy(pos);
      mesh.userData = { label: c.label, topic: c.topic };
      knowledgeGroup.add(mesh);
      knowledgeNodes[c.label] = mesh;
    });

    // Edges
    const lineMat = new THREE.LineBasicMaterial({ color: 0x10b981, transparent: true, opacity: 0.12 });
    for (const conn of connections) {
      const a = knowledgeNodes[conn.source];
      const b = knowledgeNodes[conn.target];
      if (!a || !b) continue;
      const geo = new THREE.BufferGeometry().setFromPoints([a.position, b.position]);
      const line = new THREE.Line(geo, lineMat);
      scene.add(line);
      edgeLines.push(line);
    }
  } catch (e) {
    console.warn('Failed to load knowledge graph:', e);
  }
}

// ── Stats Loading ───────────────────────────────────────────────────
async function cortexLoadStats() {
  try {
    const resp = await fetch('/api/brain/knowledge/stats');
    if (!resp.ok) return;
    const data = await resp.json();
    const el = (id) => document.getElementById(id);
    if (el('stat-topics'))       el('stat-topics').textContent = data.topics ?? '-';
    if (el('stat-concepts'))     el('stat-concepts').textContent = data.concepts ?? '-';
    if (el('stat-associations')) el('stat-associations').textContent = data.associations ?? '-';
  } catch (e) {
    console.warn('Failed to load stats:', e);
  }
}

// ── Spiking Status Polling ──────────────────────────────────────────
async function pollSpiking() {
  try {
    const resp = await fetch('/api/brain/spiking/status');
    if (!resp.ok) return;
    const data = await resp.json();
    const regions = data.regions || {};

    for (const [name, info] of Object.entries(regions)) {
      const mesh = regionMeshes[name];
      if (!mesh) continue;
      const rate = info.avg_firing_rate || 0;
      // Map firing rate (0-100 Hz) to emissive intensity (0.15 - 0.9)
      mesh.material.emissiveIntensity = 0.15 + Math.min(rate / 100, 1) * 0.75;
    }

    // Update neuron count display
    if (data.total_neurons) {
      const el = document.getElementById('neuron-count');
      if (el) el.textContent = (data.total_neurons / 1e6).toFixed(1) + 'M';
    }
  } catch (_) { /* silent */ }
}

setInterval(pollSpiking, 3000);

// ── Recall Animation ────────────────────────────────────────────────
function animateRecall(associations) {
  if (!associations || !associations.length) return;

  // Flash association cortex
  const assoc = regionMeshes['association_cortex'];
  if (assoc) {
    assoc.material.emissiveIntensity = 1.0;
    setTimeout(() => { assoc.material.emissiveIntensity = 0.15; }, 800);
  }

  // Pulse matching knowledge nodes
  for (const a of associations) {
    const label = a.label || a.concept || a;
    const node = knowledgeNodes[label];
    if (!node) continue;
    const origScale = node.scale.x;
    node.scale.setScalar(origScale * 3);
    node.material.emissiveIntensity = 1.0;
    setTimeout(() => {
      node.scale.setScalar(origScale);
      node.material.emissiveIntensity = 0.3;
    }, 600);
  }
}

// ── LOD: Shell fade + knowledge appear ──────────────────────────────
function updateLOD() {
  const dist = camera.position.length();
  // Shell fades when camera is close
  brainShell.material.opacity = THREE.MathUtils.clamp((dist - 4) / 6, 0, 0.08);
  // Knowledge nodes appear when close
  const knowledgeOpacity = THREE.MathUtils.clamp(1 - (dist - 5) / 8, 0, 1);
  knowledgeGroup.visible = knowledgeOpacity > 0.01;
  for (const child of knowledgeGroup.children) {
    child.material.opacity = knowledgeOpacity * 0.8;
  }
  for (const line of edgeLines) {
    line.material.opacity = knowledgeOpacity * 0.12;
  }
}

// ── Idle Sparkle ────────────────────────────────────────────────────
const clock = new THREE.Clock();

function idleSparkle() {
  const t = clock.getElapsedTime();
  for (const r of REGIONS) {
    const mesh = regionMeshes[r.name];
    if (!mesh) continue;
    const base = mesh.userData.baseEmissive;
    // Subtle sine modulation per region (different phase)
    const phase = r.pos[0] * 2 + r.pos[1] * 3 + r.pos[2];
    const mod = Math.sin(t * 0.8 + phase) * 0.04;
    mesh.material.emissiveIntensity = Math.max(base, mesh.material.emissiveIntensity * 0.98 + mod * 0.02);
  }
}

// ── Animation Loop ──────────────────────────────────────────────────
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  updateLOD();
  idleSparkle();
  renderer.render(scene, camera);
}
animate();

// ── Resize Handler ──────────────────────────────────────────────────
window.addEventListener('resize', () => {
  camera.aspect = width() / height();
  camera.updateProjectionMatrix();
  renderer.setSize(width(), height());
});

// ── Init ────────────────────────────────────────────────────────────
cortexLoadGraph();
cortexLoadStats();

// ── Expose to window ────────────────────────────────────────────────
window.cortexLoadGraph = cortexLoadGraph;
window.cortexLoadStats = cortexLoadStats;
window.animateRecall = animateRecall;
