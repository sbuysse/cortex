# Cortex UI Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace 17 legacy templates with a single-page immersive 3D brain explorer using Three.js, with slide-out panels, floating cards, and a unified input bar.

**Architecture:** One `index.html` template served by Axum, one `cortex.js` file for Three.js brain visualization, one `cortex.css` for custom styles. All old templates deleted. New `/api/brain/knowledge/graph` endpoint provides concept graph data for 3D rendering. Old page routes redirect to `/`.

**Tech Stack:** Three.js (CDN), Tailwind CSS (CDN), vanilla JS, Tera templates, Axum

---

## File Structure

| File | Responsibility | Action |
|------|---------------|--------|
| `templates/index.html` | Single page: top bar, Three.js canvas, panels, input bar | Create |
| `static/cortex.js` | Three.js scene: brain shell, regions, knowledge nodes, edges, animations | Create |
| `static/cortex.css` | Custom styles: panels, cards, input bar, animations | Create |
| `crates/brain-server/src/routes.rs` | New `api_brain_knowledge_graph` endpoint | Modify |
| `crates/brain-server/src/app.rs` | Single `/` route, register new API, old routes redirect | Modify |
| `templates/*.html` (17 files) | Old templates | Delete |

---

### Task 1: New API Endpoint — `/api/brain/knowledge/graph`

**Files:**
- Modify: `crates/brain-server/src/routes.rs`
- Modify: `crates/brain-server/src/app.rs`
- Modify: `crates/brain-spiking/src/knowledge.rs`

This task is backend-only and independent of the frontend.

- [ ] **Step 1: Add `graph_data` method to KnowledgeEngine**

In `crates/brain-spiking/src/knowledge.rs`, add a method that returns the full concept graph as serializable data:

```rust
    /// Export the full knowledge graph for visualization.
    /// Returns (nodes, edges) where each node has id, name, topics, degree
    /// and each edge has from_id, to_id, weight.
    pub fn graph_data(&self) -> (Vec<(usize, String, Vec<String>, usize)>, Vec<(usize, usize, f32)>) {
        let mut degree: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        for &(from, to) in self.associations.keys() {
            *degree.entry(from).or_insert(0) += 1;
            *degree.entry(to).or_insert(0) += 1;
        }

        let mut nodes = Vec::new();
        for name in self.registry.concept_names() {
            if let Some(asm) = self.registry.get(name) {
                let id = Self::concept_id(asm.start);
                let topics = self.registry.get_topics(name);
                let deg = degree.get(&id).copied().unwrap_or(0);
                nodes.push((id, name.to_string(), topics, deg));
            }
        }

        let edges: Vec<(usize, usize, f32)> = self.associations.iter()
            .map(|(&(from, to), &weight)| (from, to, weight))
            .collect();

        (nodes, edges)
    }
```

- [ ] **Step 2: Add the route handler**

In `crates/brain-server/src/routes.rs`, add:

```rust
/// Full knowledge graph for 3D visualization.
pub async fn api_brain_knowledge_graph(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    if let Some(brain) = &state.brain {
        if let Some(ref sb) = brain.spiking_brain {
            let sb = sb.lock().unwrap();
            let (nodes, edges) = sb.knowledge.graph_data();

            let json_nodes: Vec<serde_json::Value> = nodes.iter().map(|(id, name, topics, degree)| {
                serde_json::json!({
                    "id": id,
                    "name": name,
                    "topics": topics,
                    "degree": degree,
                })
            }).collect();

            let json_edges: Vec<serde_json::Value> = edges.iter().map(|(from, to, weight)| {
                serde_json::json!({
                    "from": from,
                    "to": to,
                    "weight": weight,
                })
            }).collect();

            return Json(serde_json::json!({
                "nodes": json_nodes,
                "edges": json_edges,
            })).into_response();
        }
    }
    Json(serde_json::json!({"nodes": [], "edges": []})).into_response()
}
```

- [ ] **Step 3: Register the route**

In `crates/brain-server/src/app.rs`, add after the `knowledge/stats` route:

```rust
        .route("/api/brain/knowledge/graph", get(routes::api_brain_knowledge_graph))
```

- [ ] **Step 4: Build and verify**

```bash
ssh root@prod-ia 'cd /opt/cortex/rust && cargo build --release -p brain-server'
# Then test: curl -sk https://localhost:8443/api/brain/knowledge/graph | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'{len(d[\"nodes\"])} nodes, {len(d[\"edges\"])} edges')"
```

- [ ] **Step 5: Commit**

```bash
git add crates/brain-spiking/src/knowledge.rs crates/brain-server/src/routes.rs crates/brain-server/src/app.rs
git commit -m "feat: /api/brain/knowledge/graph endpoint for 3D visualization"
```

---

### Task 2: Delete Old Templates + Create Skeleton index.html

**Files:**
- Delete: all 17 files in `templates/` (dashboard.html, goals.html, cognition.html, imagine.html, face.html, explore.html, training.html, spiking.html, brain.html, chat.html, listen.html, watch.html, youtube.html, interact.html, evolution.html, experiments.html, base.html)
- Create: `templates/index.html`
- Create: `static/cortex.css`
- Create: `static/cortex.js` (placeholder)

- [ ] **Step 1: Delete all old templates**

```bash
rm templates/dashboard.html templates/goals.html templates/cognition.html templates/imagine.html templates/face.html templates/explore.html templates/training.html templates/spiking.html templates/brain.html templates/chat.html templates/listen.html templates/watch.html templates/youtube.html templates/interact.html templates/evolution.html templates/experiments.html templates/base.html
```

- [ ] **Step 2: Create `static/` directory and `cortex.css`**

```bash
mkdir -p static
```

Create `static/cortex.css` with the foundational styles:

```css
/* Cortex UI — Immersive Brain Explorer */
* { margin: 0; padding: 0; box-sizing: border-box; }

:root {
  --bg: #050510;
  --surface: rgba(10, 10, 26, 0.95);
  --border: rgba(16, 185, 129, 0.15);
  --green: #10b981;
  --blue: #3b82f6;
  --purple: #a855f7;
  --orange: #f59e0b;
  --red: #ef4444;
  --text: #e2e8f0;
  --text-dim: #64748b;
  --text-muted: #475569;
}

body {
  background: var(--bg);
  color: var(--text);
  font-family: 'Inter', system-ui, sans-serif;
  overflow: hidden;
  height: 100vh;
  width: 100vw;
}

/* Top bar */
#top-bar {
  position: fixed; top: 0; left: 0; right: 0; height: 48px;
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  display: flex; align-items: center; padding: 0 16px;
  z-index: 100;
}
#top-bar .logo { color: var(--green); font-weight: 700; font-size: 15px; letter-spacing: 1px; }
#top-bar .stats { display: flex; gap: 6px; margin-left: 20px; }
#top-bar .stat {
  padding: 3px 10px; border-radius: 12px; font-size: 11px;
  font-family: 'JetBrains Mono', monospace;
}
#top-bar .stat.topics { background: rgba(16,185,129,0.15); color: var(--green); }
#top-bar .stat.concepts { background: rgba(59,130,246,0.15); color: var(--blue); }
#top-bar .stat.assoc { background: rgba(168,85,247,0.15); color: var(--purple); }
#top-bar .info { margin-left: auto; color: var(--text-muted); font-size: 11px; display: flex; align-items: center; gap: 8px; }
#top-bar .info .dot { width: 8px; height: 8px; background: var(--green); border-radius: 50%; }

/* Three.js canvas */
#brain-canvas {
  position: fixed; top: 48px; left: 0; right: 0; bottom: 56px;
}

/* Bottom input bar */
#input-bar {
  position: fixed; bottom: 0; left: 0; right: 0; height: 56px;
  background: var(--surface);
  border-top: 1px solid var(--border);
  display: flex; align-items: center; padding: 0 16px; gap: 12px;
  z-index: 100;
}
#input-bar input {
  flex: 1; background: #0d1117; border: 1px solid var(--border);
  border-radius: 8px; padding: 10px 16px; color: var(--text);
  font-size: 14px; font-family: 'Inter', sans-serif; outline: none;
}
#input-bar input:focus { border-color: var(--green); }
#input-bar input.loading { animation: pulse-border 1.5s ease infinite; }
#input-bar input.learning { border-color: var(--blue); }
#input-bar button {
  background: var(--green); color: #000; padding: 10px 20px;
  border-radius: 8px; border: none; font-size: 14px; font-weight: 700;
  cursor: pointer;
}
#input-bar button:hover { filter: brightness(1.1); }

/* Slide-out panels */
.panel-tabs {
  position: fixed; top: 60px; left: 0; display: flex; flex-direction: column;
  gap: 4px; z-index: 90;
}
.panel-tab {
  background: var(--surface); border: 1px solid var(--border);
  border-left: none; border-radius: 0 8px 8px 0;
  padding: 10px 6px; font-size: 10px; color: var(--text-dim);
  writing-mode: vertical-rl; cursor: pointer; letter-spacing: 1px;
  transition: background 0.2s;
}
.panel-tab:hover, .panel-tab.active { background: rgba(16,185,129,0.1); color: var(--green); }
.panel-tab.knowledge { border-color: rgba(16,185,129,0.2); }
.panel-tab.regions { border-color: rgba(59,130,246,0.2); }
.panel-tab.stats { border-color: rgba(168,85,247,0.2); }

.slide-panel {
  position: fixed; top: 48px; left: 0; bottom: 56px; width: 300px;
  background: var(--surface); border-right: 1px solid var(--border);
  padding: 16px; overflow-y: auto; z-index: 80;
  transform: translateX(-100%); transition: transform 0.3s ease;
}
.slide-panel.open { transform: translateX(0); }
.slide-panel h3 { color: var(--green); font-size: 11px; letter-spacing: 1px; margin-bottom: 12px; text-transform: uppercase; }
.slide-panel .item {
  padding: 8px; border-radius: 6px; cursor: pointer; font-size: 13px;
  color: var(--text-dim); transition: background 0.15s;
}
.slide-panel .item:hover { background: rgba(16,185,129,0.1); color: var(--text); }
.slide-panel .item .meta { font-size: 11px; color: var(--text-muted); }

/* Floating cards */
.float-card {
  position: fixed; background: var(--surface);
  border: 1px solid var(--border); border-radius: 12px;
  padding: 16px; z-index: 70; max-width: 380px;
  box-shadow: 0 8px 32px rgba(0,0,0,0.5);
  animation: slide-in 0.3s ease;
}
.float-card .card-title { font-size: 10px; letter-spacing: 1px; margin-bottom: 8px; font-weight: 700; }
.float-card .card-body { font-size: 13px; color: #ccc; line-height: 1.6; }
.float-card .card-tags { border-top: 1px solid rgba(255,255,255,0.1); padding-top: 8px; margin-top: 8px; font-size: 11px; }
.float-card .close-btn {
  position: absolute; top: 8px; right: 10px; background: none;
  border: none; color: var(--text-muted); cursor: pointer; font-size: 16px;
}

.tag-confirmed { color: var(--green); }
.tag-explicit { color: var(--blue); }
.tag-emergent { color: var(--purple); }
.tag-predicted { color: var(--orange); }

/* Animations */
@keyframes pulse-border { 0%,100% { border-color: var(--green); } 50% { border-color: transparent; } }
@keyframes slide-in { from { opacity: 0; transform: translateX(20px); } to { opacity: 1; transform: translateX(0); } }
@keyframes fade-in { from { opacity: 0; } to { opacity: 1; } }
@keyframes pop-in { from { transform: scale(0); } to { transform: scale(1); } }
```

- [ ] **Step 3: Create `static/cortex.js` placeholder**

```javascript
// Cortex 3D Brain Visualization — placeholder, built in Task 3
console.log('Cortex UI loading...');
```

- [ ] **Step 4: Create `templates/index.html`**

Create the complete single-page template:

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Cortex — Spiking Brain Explorer</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="/static/cortex.css">
</head>
<body>

  <!-- Top bar -->
  <div id="top-bar">
    <span class="logo">CORTEX</span>
    <div class="stats">
      <span class="stat topics" id="stat-topics">– topics</span>
      <span class="stat concepts" id="stat-concepts">– concepts</span>
      <span class="stat assoc" id="stat-assoc">– associations</span>
    </div>
    <div class="info">
      <span class="dot"></span>
      <span id="stat-neurons">2M neurons · 2B synapses</span>
    </div>
  </div>

  <!-- Three.js canvas -->
  <div id="brain-canvas"></div>

  <!-- Panel tabs (left edge) -->
  <div class="panel-tabs">
    <div class="panel-tab knowledge" onclick="togglePanel('knowledge')">Knowledge</div>
    <div class="panel-tab regions" onclick="togglePanel('regions')">Regions</div>
    <div class="panel-tab stats" onclick="togglePanel('stats')">Stats</div>
  </div>

  <!-- Knowledge panel -->
  <div class="slide-panel" id="panel-knowledge">
    <h3>Knowledge Browser</h3>
    <input type="text" placeholder="Search concepts..." style="width:100%; background:#0d1117; border:1px solid rgba(16,185,129,0.2); border-radius:6px; padding:8px; color:#e2e8f0; font-size:13px; margin-bottom:12px; outline:none;">
    <div id="knowledge-list"></div>
  </div>

  <!-- Regions panel -->
  <div class="slide-panel" id="panel-regions">
    <h3>Brain Regions</h3>
    <div id="regions-list"></div>
  </div>

  <!-- Stats panel -->
  <div class="slide-panel" id="panel-stats">
    <h3>System Stats</h3>
    <div id="stats-detail"></div>
  </div>

  <!-- Bottom input bar -->
  <div id="input-bar">
    <input type="text" id="main-input" placeholder="Ask a question, paste a YouTube URL, or type /help..."
           onkeydown="if(event.key==='Enter')handleInput()">
    <button onclick="handleInput()">↵</button>
  </div>

  <!-- Three.js -->
  <script src="https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/controls/OrbitControls.js" type="module"></script>
  <script src="/static/cortex.js" type="module"></script>

</body>
</html>
```

- [ ] **Step 5: Update `app.rs` — single `/` route, redirect old pages**

Replace the HTML page route section in `crates/brain-server/src/app.rs`. Change lines 15-32 from individual page routes to:

```rust
        // Single-page UI
        .route("/", get(routes::index_page))
        // Old URLs redirect to /
        .route("/goals", get(routes::redirect_home))
        .route("/cognition", get(routes::redirect_home))
        .route("/imagine", get(routes::redirect_home))
        .route("/face", get(routes::redirect_home))
        .route("/explore", get(routes::redirect_home))
        .route("/training", get(routes::redirect_home))
        .route("/spiking", get(routes::redirect_home))
        .route("/evolution", get(routes::redirect_home))
        .route("/experiments", get(routes::redirect_home))
        .route("/interact", get(routes::redirect_home))
        .route("/brain", get(routes::redirect_home))
        .route("/youtube", get(routes::redirect_home))
        .route("/chat", get(routes::redirect_home))
        .route("/listen", get(routes::redirect_home))
        .route("/watch", get(routes::redirect_home))
```

- [ ] **Step 6: Add `index_page` and `redirect_home` route handlers**

In `crates/brain-server/src/routes.rs`, add:

```rust
/// Single-page brain explorer.
pub async fn index_page(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let ctx = tera::Context::new();
    match state.templates.render("index.html", &ctx) {
        Ok(html) => Html(html).into_response(),
        Err(e) => Html(format!("Template error: {e:?}")).into_response(),
    }
}

/// Redirect old pages to /.
pub async fn redirect_home() -> impl IntoResponse {
    axum::response::Redirect::permanent("/")
}
```

- [ ] **Step 7: Remove old page handler references**

Remove the old handler functions (`dashboard`, `goals`, `cognition`, `imagine`, `face`, `explore`, `training`, `spiking_page`) from routes.rs. Search for each and delete the function body. Keep all `/api/` handlers unchanged.

Note: some old handlers may be referenced by the catch-all backward compat routes. After changing those to `redirect_home`, the old handlers are dead code.

- [ ] **Step 8: Build, deploy, verify skeleton loads**

```bash
rsync -az templates/ root@prod-ia:/opt/cortex/templates/
rsync -az static/ root@prod-ia:/opt/cortex/static/
rsync -az rust/ root@prod-ia:/opt/cortex/rust/
ssh root@prod-ia 'cd /opt/cortex/rust && cargo build --release -p brain-server'
# Restart and open https://prod-ia:8443/ — should see the skeleton with top bar + input bar
```

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "feat: delete 17 old templates, create single-page brain explorer skeleton"
```

---

### Task 3: Three.js Brain Visualization — Shell + Regions

**Files:**
- Modify: `static/cortex.js`

This is the core 3D rendering. Builds incrementally — first the brain shell and regions, then knowledge nodes in Task 4.

- [ ] **Step 1: Write the complete cortex.js with brain shell + regions**

Replace `static/cortex.js` with the full Three.js brain visualization. The file should:

1. Create a Three.js scene with dark background
2. Add OrbitControls for mouse rotation/zoom
3. Create a transparent brain shell (ellipsoid wireframe)
4. Create 10 region meshes positioned anatomically
5. Poll `/api/brain/spiking/status` every 3s to update region glow intensity
6. Load `/api/brain/knowledge/stats` on startup to populate top bar stats

The file is a module (`type="module"` in the HTML). Use Three.js from CDN via importmap or direct import.

```javascript
// Cortex 3D Brain Visualization
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js';
import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/controls/OrbitControls.js';

// ── Scene setup ──
const container = document.getElementById('brain-canvas');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x050510);

const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, 2, 8);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setSize(window.innerWidth, window.innerHeight - 104); // minus top+bottom bars
renderer.setPixelRatio(window.devicePixelRatio);
container.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.3;

// ── Lighting ──
scene.add(new THREE.AmbientLight(0x404060, 0.5));
const pointLight = new THREE.PointLight(0x10b981, 1, 20);
pointLight.position.set(5, 5, 5);
scene.add(pointLight);

// ── Brain shell (transparent wireframe ellipsoid) ──
const shellGeom = new THREE.SphereGeometry(3, 32, 24);
shellGeom.scale(1.2, 1, 1); // ellipsoid
const shellMat = new THREE.MeshBasicMaterial({
  color: 0x10b981, wireframe: true, transparent: true, opacity: 0.08
});
const brainShell = new THREE.Mesh(shellGeom, shellMat);
scene.add(brainShell);

// ── Brain regions ──
const REGIONS = [
  { name: 'visual_cortex',     pos: [0, 0.5, -2.2],   size: [0.8, 0.5, 0.5], color: 0x3b82f6 },
  { name: 'auditory_cortex',   pos: [2, 0, -0.5],      size: [0.5, 0.4, 0.6], color: 0xf59e0b },
  { name: 'association_cortex',pos: [0, 1, 0],          size: [1.5, 0.8, 1.2], color: 0x10b981 },
  { name: 'predictive_cortex', pos: [0, 1.5, -1],       size: [0.7, 0.4, 0.5], color: 0x06b6d4 },
  { name: 'hippocampus',       pos: [0, -0.3, 0],       size: [0.9, 0.5, 0.6], color: 0xa855f7 },
  { name: 'prefrontal_cortex', pos: [0, 0.8, 2],        size: [0.8, 0.6, 0.5], color: 0xef4444 },
  { name: 'amygdala',          pos: [0, -0.8, 0.5],     size: [0.4, 0.4, 0.4], color: 0xf97316 },
  { name: 'motor_cortex',      pos: [0, 2, 1],          size: [0.7, 0.3, 0.5], color: 0x14b8a6 },
  { name: 'brainstem',         pos: [0, -2, 0],         size: [0.4, 0.6, 0.4], color: 0x8b5cf6 },
  { name: 'cerebellum',        pos: [0, -1.5, -1.5],    size: [1, 0.5, 0.7],   color: 0x22d3ee },
];

const regionMeshes = {};
REGIONS.forEach(r => {
  const geom = new THREE.SphereGeometry(1, 16, 12);
  geom.scale(...r.size);
  const mat = new THREE.MeshPhongMaterial({
    color: r.color, transparent: true, opacity: 0.15,
    emissive: r.color, emissiveIntensity: 0.1,
  });
  const mesh = new THREE.Mesh(geom, mat);
  mesh.position.set(...r.pos);
  mesh.userData = { name: r.name, baseEmissive: 0.1 };
  scene.add(mesh);
  regionMeshes[r.name] = mesh;
});

// ── Knowledge nodes (placeholder group, populated in Task 4) ──
const knowledgeGroup = new THREE.Group();
scene.add(knowledgeGroup);
window.cortexKnowledgeGroup = knowledgeGroup;

// ── State ──
let graphData = { nodes: [], edges: [] };
let stats = {};

// ── Data loading ──
async function loadStats() {
  try {
    const res = await fetch('/api/brain/knowledge/stats');
    stats = await res.json();
    document.getElementById('stat-topics').textContent = `${stats.total_topics || 0} topics`;
    document.getElementById('stat-concepts').textContent = `${stats.total_concepts || 0} concepts`;
    document.getElementById('stat-assoc').textContent = `${stats.total_associations || 0} associations`;
  } catch(e) { console.warn('Stats load failed:', e); }
}

async function loadGraph() {
  try {
    const res = await fetch('/api/brain/knowledge/graph');
    graphData = await res.json();
    window.cortexGraphData = graphData;
    buildKnowledgeNodes();
  } catch(e) { console.warn('Graph load failed:', e); }
}

async function pollRegions() {
  try {
    const res = await fetch('/api/brain/spiking/status');
    const data = await res.json();
    if (data.regions) {
      data.regions.forEach(r => {
        const mesh = regionMeshes[r.name];
        if (mesh) {
          const intensity = Math.min(r.spike_rate * 50 + 0.1, 1.0);
          mesh.material.emissiveIntensity = intensity;
          mesh.material.opacity = 0.15 + intensity * 0.3;
        }
      });
    }
    // Update neuron stats
    if (data.total_neurons) {
      const m = (data.total_neurons / 1e6).toFixed(1);
      const b = (data.total_synapses / 1e9).toFixed(1);
      document.getElementById('stat-neurons').textContent = `${m}M neurons · ${b}B synapses`;
    }
  } catch(e) {}
}

// ── Knowledge node rendering ──
function buildKnowledgeNodes() {
  // Clear existing
  while (knowledgeGroup.children.length) knowledgeGroup.remove(knowledgeGroup.children[0]);

  if (!graphData.nodes || graphData.nodes.length === 0) return;

  // Topic color map
  const topicColors = {};
  const hues = [0.33, 0.6, 0.8, 0.1, 0.45, 0.75, 0.15, 0.55, 0.9, 0.05];
  let hueIdx = 0;

  // Position nodes within association cortex region (center of brain)
  const assocCenter = new THREE.Vector3(0, 1, 0);

  graphData.nodes.forEach((node, i) => {
    // Assign color by first topic
    const topic = (node.topics && node.topics[0]) || 'unknown';
    if (!topicColors[topic]) {
      topicColors[topic] = new THREE.Color().setHSL(hues[hueIdx % hues.length], 0.7, 0.6);
      hueIdx++;
    }

    // Position: hash-based within a sphere around association cortex
    const phi = (node.id * 2.399) % (Math.PI * 2); // golden angle
    const cosTheta = 1 - 2 * ((node.id * 0.618) % 1);
    const sinTheta = Math.sqrt(1 - cosTheta * cosTheta);
    const r = 0.8 + (node.degree || 1) * 0.02;

    const pos = new THREE.Vector3(
      assocCenter.x + r * sinTheta * Math.cos(phi),
      assocCenter.y + r * cosTheta * 0.6,
      assocCenter.z + r * sinTheta * Math.sin(phi)
    );

    const size = 0.02 + Math.min((node.degree || 0) * 0.003, 0.08);
    const geom = new THREE.SphereGeometry(size, 8, 6);
    const mat = new THREE.MeshBasicMaterial({
      color: topicColors[topic], transparent: true, opacity: 0.8,
    });
    const mesh = new THREE.Mesh(geom, pos);
    mesh.position.copy(pos);
    mesh.userData = { nodeId: node.id, name: node.name, topics: node.topics, degree: node.degree };
    knowledgeGroup.add(mesh);
  });

  // Edges as lines
  const edgeMat = new THREE.LineBasicMaterial({ color: 0x10b981, transparent: true, opacity: 0.1 });
  const nodeMap = {};
  knowledgeGroup.children.forEach(m => { if (m.userData.nodeId !== undefined) nodeMap[m.userData.nodeId] = m; });

  graphData.edges.forEach(edge => {
    const fromMesh = nodeMap[edge.from];
    const toMesh = nodeMap[edge.to];
    if (fromMesh && toMesh) {
      const points = [fromMesh.position.clone(), toMesh.position.clone()];
      const geom = new THREE.BufferGeometry().setFromPoints(points);
      const opacity = Math.min(edge.weight * 0.3, 0.4);
      const mat = new THREE.LineBasicMaterial({ color: 0x10b981, transparent: true, opacity });
      const line = new THREE.Line(geom, mat);
      knowledgeGroup.add(line);
    }
  });
}

// ── Zoom LOD: fade shell, show nodes ──
function updateLOD() {
  const dist = camera.position.length();
  // Shell: visible at distance, fades when close
  brainShell.material.opacity = Math.max(0, Math.min(0.08, (dist - 4) * 0.02));
  // Regions: fade when very close
  Object.values(regionMeshes).forEach(m => {
    const baseOpacity = 0.15 + (m.material.emissiveIntensity || 0.1) * 0.3;
    m.material.opacity = dist < 3 ? baseOpacity * ((dist - 1) / 2) : baseOpacity;
  });
  // Knowledge nodes: visible when close
  knowledgeGroup.visible = dist < 7;
  knowledgeGroup.children.forEach(c => {
    if (c.material) c.material.opacity = Math.min(1, Math.max(0, (7 - dist) * 0.3));
  });
}

// ── Resize handler ──
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / (window.innerHeight - 104);
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight - 104);
});

// ── Animation loop ──
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  updateLOD();
  renderer.render(scene, camera);
}

// ── Init ──
loadStats();
loadGraph();
setInterval(pollRegions, 3000);
setInterval(loadStats, 10000);
animate();

// ── Expose for input handler ──
window.cortexScene = scene;
window.cortexCamera = camera;
window.cortexRegionMeshes = regionMeshes;
window.cortexLoadGraph = loadGraph;
window.cortexLoadStats = loadStats;
```

- [ ] **Step 2: Add inline JavaScript for panels + input handler to index.html**

Add before the closing `</body>` in `templates/index.html`, after the Three.js script tags:

```html
<script>
  // ── Panel toggle ──
  let activePanel = null;
  function togglePanel(name) {
    document.querySelectorAll('.slide-panel').forEach(p => p.classList.remove('open'));
    document.querySelectorAll('.panel-tab').forEach(t => t.classList.remove('active'));
    if (activePanel === name) { activePanel = null; return; }
    document.getElementById('panel-' + name).classList.add('open');
    document.querySelector('.panel-tab.' + name).classList.add('active');
    activePanel = name;
    if (name === 'knowledge') loadKnowledgePanel();
    if (name === 'regions') loadRegionsPanel();
    if (name === 'stats') loadStatsPanel();
  }

  async function loadKnowledgePanel() {
    const res = await fetch('/api/brain/knowledge/stats');
    const data = await res.json();
    const list = document.getElementById('knowledge-list');
    let html = '<div style="margin-bottom:16px;"><div style="color:#888;font-size:11px;margin-bottom:6px;">TOPICS (' + (data.total_topics||0) + ')</div>';
    (data.topic_details || []).forEach(t => {
      html += '<div class="item">' + t.topic + ' <span class="meta">' + (t.triples_count||0) + ' triples</span></div>';
    });
    html += '</div><div><div style="color:#f59e0b;font-size:11px;margin-bottom:6px;">BRIDGE CONCEPTS</div>';
    (data.bridge_concepts || []).forEach(b => {
      html += '<div class="item" style="color:#f59e0b;">' + b + '</div>';
    });
    html += '</div>';
    list.innerHTML = html;
  }

  async function loadRegionsPanel() {
    const res = await fetch('/api/brain/spiking/status');
    const data = await res.json();
    const list = document.getElementById('regions-list');
    let html = '';
    (data.regions || []).forEach(r => {
      const rate = (r.spike_rate * 100).toFixed(2);
      html += '<div class="item"><strong>' + r.name.replace(/_/g,' ') + '</strong><br><span class="meta">' +
        r.neurons.toLocaleString() + ' neurons · ' + r.synapses.toLocaleString() + ' synapses · ' + rate + '% spike rate</span></div>';
    });
    list.innerHTML = html;
  }

  async function loadStatsPanel() {
    const res = await fetch('/api/brain/knowledge/stats');
    const data = await res.json();
    const el = document.getElementById('stats-detail');
    el.innerHTML =
      '<div class="item">Topics: <strong>' + (data.total_topics||0) + '</strong></div>' +
      '<div class="item">Concepts: <strong>' + (data.total_concepts||0) + '</strong></div>' +
      '<div class="item">Associations: <strong>' + (data.total_associations||0) + '</strong></div>' +
      '<div class="item">Bridge concepts: <strong>' + (data.bridge_concepts||[]).length + '</strong></div>' +
      '<div class="item" style="margin-top:12px;"><div style="color:#888;font-size:11px;margin-bottom:4px;">TOP CONNECTED</div>' +
      (data.top_connected||[]).map(c => '<div style="font-size:12px;color:#aaa;">' + c.concept + ' <span class="meta">' + c.degree + ' edges</span></div>').join('') +
      '</div>';
  }

  // ── Input handler ──
  async function handleInput() {
    const input = document.getElementById('main-input');
    const text = input.value.trim();
    if (!text) return;
    input.value = '';

    if (text.startsWith('http')) {
      // YouTube URL — learn
      input.classList.add('learning');
      showCard('learning', { topic: 'Learning...', progress: 0 });
      try {
        const topic = text.split('v=')[1] ? 'YouTube video' : 'video';
        const topicPrompt = prompt('Topic name for this video:', topic);
        const res = await fetch('/api/brain/learn/academic', {
          method: 'POST', headers: {'Content-Type':'application/json'},
          body: JSON.stringify({ query: text, topic: topicPrompt || topic })
        });
        const data = await res.json();
        showCard('learning', { topic: topicPrompt, progress: 100, triples: data.concepts_learned });
        setTimeout(() => { hideCard('learning'); if(window.cortexLoadGraph) window.cortexLoadGraph(); if(window.cortexLoadStats) window.cortexLoadStats(); }, 3000);
      } catch(e) { showCard('learning', { topic: 'Error', error: e.message }); }
      input.classList.remove('learning');

    } else if (text.startsWith('/')) {
      // Command
      const cmd = text.slice(1).split(' ')[0];
      if (cmd === 'help') showCard('response', { text: 'Commands: /topics, /stats, /regions, /help\nOr ask any question. Paste a YouTube URL to learn.' });
      else if (cmd === 'topics') togglePanel('knowledge');
      else if (cmd === 'stats') togglePanel('stats');
      else if (cmd === 'regions') togglePanel('regions');
      else showCard('response', { text: 'Unknown command: /' + cmd });

    } else {
      // Question — ask brain
      input.classList.add('loading');
      try {
        // Prime
        await fetch('/api/brain/dialogue/grounded', {
          method: 'POST', headers: {'Content-Type':'application/json'},
          body: JSON.stringify({ message: text, session_id: 'ui-' + Date.now() })
        });
        // Wait for tick
        await new Promise(r => setTimeout(r, 5000));
        // Ask
        const res = await fetch('/api/brain/dialogue/grounded', {
          method: 'POST', headers: {'Content-Type':'application/json'},
          body: JSON.stringify({ message: text, session_id: 'ui-' + Date.now() })
        });
        const data = await res.json();
        showCard('response', {
          text: data.response || 'No response',
          associations: data.brain_associations || []
        });
      } catch(e) { showCard('response', { text: 'Error: ' + e.message }); }
      input.classList.remove('loading');
    }
  }

  // ── Floating cards ──
  function showCard(type, data) {
    hideCard(type);
    const card = document.createElement('div');
    card.className = 'float-card';
    card.id = 'card-' + type;

    if (type === 'response') {
      card.style.cssText = 'top:70px;right:20px;';
      let tagsHtml = '';
      if (data.associations && data.associations.length > 1) {
        tagsHtml = '<div class="card-tags">' + data.associations
          .filter(a => a.startsWith('['))
          .map(a => {
            const tag = a.match(/\[(\w+)\]/)?.[1] || '';
            return '<span class="tag-' + tag + '">' + a + '</span>';
          }).join(' · ') + '</div>';
      }
      card.innerHTML = '<button class="close-btn" onclick="hideCard(\'response\')">&times;</button>' +
        '<div class="card-title" style="color:var(--green);">CORTEX RESPONSE</div>' +
        '<div class="card-body">' + data.text + '</div>' + tagsHtml;

    } else if (type === 'learning') {
      card.style.cssText = 'bottom:70px;right:20px;width:280px;';
      card.innerHTML = '<button class="close-btn" onclick="hideCard(\'learning\')">&times;</button>' +
        '<div class="card-title" style="color:var(--blue);">LEARNING</div>' +
        '<div class="card-body">' + (data.topic || '') +
        (data.triples ? ' · ' + data.triples + ' concepts' : '') +
        (data.error ? '<br><span style="color:var(--red);">' + data.error + '</span>' : '') +
        '</div>' +
        '<div style="background:#1a1a2e;border-radius:4px;height:4px;margin-top:8px;"><div style="background:var(--blue);border-radius:4px;height:4px;width:' + (data.progress||0) + '%;transition:width 0.5s;"></div></div>';
    }
    document.body.appendChild(card);
  }

  function hideCard(type) {
    const el = document.getElementById('card-' + type);
    if (el) el.remove();
  }
</script>
```

- [ ] **Step 3: Deploy and test in browser**

```bash
rsync -az static/ root@prod-ia:/opt/cortex/static/
rsync -az templates/ root@prod-ia:/opt/cortex/templates/
rsync -az rust/ root@prod-ia:/opt/cortex/rust/
ssh root@prod-ia 'cd /opt/cortex/rust && cargo build --release -p brain-server && kill $(pgrep brain-server); sleep 2; cd /opt/cortex && bash start.sh > brain.log 2>&1 &'
```

Open `https://prod-ia:8443/` — should see:
- Rotating 3D brain with 10 colored regions
- Knowledge nodes as dots inside the brain
- Top bar with live stats
- Bottom input bar
- Working slide-out panels
- Ask a question → floating response card

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat: immersive 3D brain explorer — Three.js, panels, unified input, floating cards"
```

---

### Task 4: Polish and Real-Time Animations

**Files:**
- Modify: `static/cortex.js`
- Modify: `static/cortex.css`

- [ ] **Step 1: Add recall animation to cortex.js**

Add a function that highlights regions and nodes during recall. Append to `cortex.js`:

```javascript
// ── Recall animation ──
window.animateRecall = function(associations) {
  // Flash association cortex
  const assoc = regionMeshes['association_cortex'];
  if (assoc) {
    assoc.material.emissiveIntensity = 1.0;
    setTimeout(() => { assoc.material.emissiveIntensity = 0.3; }, 2000);
  }

  // Highlight matching knowledge nodes
  if (!window.cortexKnowledgeGroup) return;
  const names = associations.filter(a => a.startsWith('[')).map(a => a.replace(/\[\w+\]\s*/, '').trim().toLowerCase());

  window.cortexKnowledgeGroup.children.forEach(child => {
    if (child.userData && child.userData.name && names.includes(child.userData.name.toLowerCase())) {
      child.material.opacity = 1.0;
      child.scale.setScalar(2.5);
      setTimeout(() => { child.scale.setScalar(1); child.material.opacity = 0.8; }, 3000);
    }
  });
};
```

- [ ] **Step 2: Wire animation into the input handler**

In `templates/index.html`, update the question handler's response section. After `showCard('response', ...)`, add:

```javascript
        if (window.animateRecall && data.brain_associations) {
          window.animateRecall(data.brain_associations);
        }
```

- [ ] **Step 3: Add idle sparkle animation to cortex.js**

Append to `cortex.js`:

```javascript
// ── Idle sparkle ──
let sparkleTime = 0;
function updateSparkle(delta) {
  sparkleTime += delta;
  Object.values(regionMeshes).forEach((mesh, i) => {
    const base = mesh.userData.baseEmissive || 0.1;
    const sparkle = Math.sin(sparkleTime * 0.5 + i * 1.3) * 0.03;
    mesh.material.emissiveIntensity = Math.max(base, mesh.material.emissiveIntensity * 0.99 + sparkle);
  });
}
```

Update the animate loop to include sparkle:

```javascript
const clock = new THREE.Clock();
function animate() {
  requestAnimationFrame(animate);
  const delta = clock.getDelta();
  controls.update();
  updateLOD();
  updateSparkle(delta);
  renderer.render(scene, camera);
}
```

- [ ] **Step 4: Deploy and verify animations**

```bash
rsync -az static/ root@prod-ia:/opt/cortex/static/
rsync -az templates/ root@prod-ia:/opt/cortex/templates/
```

Open browser, ask a question — association cortex should flash bright, matching concept nodes should pulse larger.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: recall animation + idle sparkle for 3D brain"
```

---

### Task 5: Deploy Final + Update README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Full sync and restart**

```bash
rsync -az /home/sbuysse/Documents/Coding/Projects/Akretio/Brain/ root@prod-ia:/opt/cortex/ --exclude=.git --exclude=target
ssh root@prod-ia 'cd /opt/cortex/rust && cargo build --release -p brain-server && kill $(pgrep brain-server); sleep 2; cd /opt/cortex && bash start.sh > brain.log 2>&1 &'
```

- [ ] **Step 2: Update README**

Add a UI section to README.md after the Performance table:

```markdown
## UI — Immersive Brain Explorer

Open `https://your-server:8443/` to access the 3D brain explorer.

- **Full-screen 3D brain** with 10 anatomically positioned regions that glow based on spike activity
- **Knowledge graph** visible when zoomed in — 1000+ concept nodes colored by topic, connected by learned associations
- **Ask questions** via the unified input bar — the brain animates during recall, response appears as a floating card with confidence tags
- **Learn from YouTube** — paste a URL, the brain learns in real-time with progress animation
- **Browse knowledge** — slide-out panels for topics, brain regions, and system stats
- **Confidence visualization** — [confirmed] green, [explicit] blue, [emergent] purple, [predicted] orange

Built with Three.js, vanilla JS, and Tailwind CSS. No framework required.
```

- [ ] **Step 3: Push**

```bash
git add -A
git commit -m "feat: v0.5.0 — immersive 3D brain explorer UI"
git push origin master
```
