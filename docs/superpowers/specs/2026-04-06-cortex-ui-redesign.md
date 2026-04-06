# Cortex UI Redesign — Immersive Brain Explorer

## Goal

Replace the 17-template legacy UI with a single-page immersive brain explorer. Full-screen 3D brain visualization with slide-out panels, unified input bar, and real-time animations during learning and recall. Start from scratch — delete all old templates.

## Audience

- **Demo visitors / GitHub:** Impressive 3D brain that shows what Cortex can do
- **Power users (developer):** Slide-out panels for knowledge browsing, region stats, and detailed concept inspection

## Tech Stack

- **Three.js** — 3D brain rendering (transparent shell, region meshes, knowledge graph nodes)
- **Vanilla JS** — no framework, same approach as current UI
- **Tailwind CSS** — utility styling (CDN)
- **SSE** — real-time updates from brain (spike activity, learning progress)
- **Tera templates** — served by Axum (single `index.html` template)

## Architecture

```
Single page: index.html
  ├── Top bar: CORTEX logo + live stats (topics, concepts, associations, neurons)
  ├── 3D viewport (Three.js canvas, fills screen)
  │   ├── Brain shell (transparent sphere/ellipsoid)
  │   ├── 10 region meshes (positioned anatomically, glow by activity)
  │   ├── Knowledge nodes (dots within association cortex, colored by topic)
  │   ├── Connection edges (lines between nodes, colored by type)
  │   └── Zoom: shell fades → knowledge graph visible → click node to inspect
  ├── Slide-out panels (left edge tabs)
  │   ├── Knowledge: topic list, concept search, bridge concepts
  │   ├── Regions: 10 regions with neuron count, spike rate, synapse count
  │   └── Stats: total metrics, learning history, recall performance
  ├── Floating cards (appear contextually)
  │   ├── Response card: answer + confidence tags after asking
  │   ├── Learning card: progress during YouTube learning
  │   └── Concept card: details when clicking a node
  └── Bottom bar: unified input (question / URL / command)
```

## Pages

**One page only: `index.html`** — replaces all 17 existing templates. Old routes redirect to `/`.

## Section 1: 3D Brain Visualization (Three.js)

### Brain Shell
- Transparent ellipsoid mesh (~brain shape), wireframe or glass material
- Rotates slowly by default (can be paused by mouse interaction)
- Mouse drag to rotate, scroll to zoom, right-click to pan

### Brain Regions (10 meshes)
- Each region is a soft ellipsoid positioned anatomically within the shell:
  - Visual cortex: back
  - Auditory cortex: sides
  - Association cortex: center-top (largest)
  - Hippocampus: center-inner
  - PFC: front
  - Amygdala: center-lower
  - Motor cortex: top-front
  - Predictive cortex: upper-back
  - Brainstem: bottom-center
  - Cerebellum: bottom-back
- Color: each region has a base color, intensity modulated by spike rate (from `/api/brain/spiking/status`)
- Hover: tooltip with region name, neuron count, spike rate

### Knowledge Nodes (within association cortex)
- Each registered concept = one small sphere positioned within the association cortex mesh
- Position: derived from concept_id (hash to 3D coordinates within the region bounds)
- Color: by topic (each topic gets a unique hue)
- Size: by connection degree (top_connected concepts are bigger)
- Visible when zoomed in past a threshold (LOD: shell at distance, nodes when close)

### Connection Edges
- Lines between connected concept nodes
- Color by confidence tag: green = confirmed, blue = explicit, purple = emergent, orange = predicted
- Opacity by weight (stronger = more opaque)
- Animate: flash/pulse when a connection is activated during recall

### Zoom Transition
- Far: see brain shell with glowing regions
- Mid: shell becomes translucent, region meshes visible
- Close: shell disappears, knowledge graph nodes and edges visible
- Click a node: floating concept detail card appears

### Data Source
- Periodic polling of `/api/brain/spiking/status` (every 2s) for region activity
- One-time load of `/api/brain/knowledge/stats` for concept positions and connections
- SSE stream for real-time events (learning progress, recall results)

## Section 2: Slide-Out Panels

Three panels accessible via tabs on the left edge of the screen.

### Knowledge Panel
- **Topic list:** All learned topics with triple count, click to highlight that topic's concepts in the 3D view
- **Concept search:** Type to filter, click to zoom to that concept in the brain
- **Bridge concepts:** Concepts shared by 2+ topics, with topic tags

### Regions Panel
- **10 region cards:** Name, neuron count, synapse count, current spike rate, learning enabled status
- Click region card → camera flies to that region in the 3D view

### Stats Panel
- Total topics, concepts, associations, persisted triples
- Brain scale: neurons, synapses, regions
- Recall timing: BFS (ms), spiking (ms)
- Learning history: last 10 learned topics with timestamps

## Section 3: Floating Cards

Cards appear contextually over the 3D view and can be dismissed.

### Response Card (after asking)
- Shows LLM response text
- Below: confidence-tagged associations `[confirmed] kv cache · [emergent] self-attention · [predicted] sparse`
- Click a tagged concept → highlights it in the 3D brain, draws connections

### Learning Card (during YouTube learning)
- Shows topic name, extraction progress
- Triple count as it grows
- Synapse imprint count
- Progress bar
- Dismisses when learning completes

### Concept Detail Card (when clicking a node)
- Concept name
- Topics it was learned from
- Connected concepts (with edge type and weight)
- How many synapses imprinted for this concept

## Section 4: Unified Input Bar

Single input field at the bottom of the screen.

### Smart Detection
- Starts with `http` → treat as YouTube URL → trigger learning
- Starts with `/` → command (see below)
- Otherwise → question → trigger recall + LLM response

### Commands
- `/help` — show available commands
- `/stats` — toggle stats panel
- `/topics` — toggle knowledge panel
- `/regions` — toggle regions panel
- `/learn <url> <topic>` — learn from URL with explicit topic
- `/batch <url1> <topic1> <url2> <topic2> ...` — batch learn

### Visual Feedback
- While processing: input bar pulses green
- During learning: bar turns blue with progress
- On error: bar flashes red briefly

## Section 5: Real-Time Animations

### During Recall
1. Input submitted → input bar pulses
2. Association cortex region glows brighter (BFS running)
3. Seed concept nodes flash white
4. Spike propagation: waves of light spread from seed nodes through connections
5. Activated concept nodes glow (green = confirmed, purple = emergent, orange = predicted)
6. Response card slides in from the right

### During Learning
1. URL submitted → input bar turns blue
2. Learning card appears with progress
3. As triples are extracted: new concept nodes appear with a pop animation
4. As synapses are imprinted: connection edges animate in
5. Brain regions pulse briefly
6. Learning card shows "Complete" then fades

### Idle
- Brain shell rotates slowly
- Occasional subtle sparkle on active regions
- Region brightness reflects baseline spike activity

## Section 6: API Endpoints Used

| Endpoint | Purpose | Frequency |
|----------|---------|-----------|
| `GET /api/brain/spiking/status` | Region activity, spike rates | Poll every 2s |
| `GET /api/brain/knowledge/stats` | Topics, concepts, bridges, top connected | On load + after learning |
| `POST /api/brain/dialogue/grounded` | Ask question, get response + associations | On user query |
| `POST /api/brain/learn/academic` | Learn from YouTube URL | On user URL input |
| `POST /api/brain/learn/batch` | Batch learn multiple URLs | On /batch command |
| `GET /api/brain/knowledge/graph` | **NEW** — Full concept graph (nodes + edges) for 3D rendering | On load + after learning |

### New Endpoint: `/api/brain/knowledge/graph`

Returns the full knowledge graph in a format optimized for Three.js rendering:

```json
{
  "nodes": [
    {"id": 0, "name": "kv cache", "topics": ["TurboQuant", "FlashAttention"], "degree": 12},
    {"id": 1, "name": "quantization", "topics": ["TurboQuant"], "degree": 8}
  ],
  "edges": [
    {"from": 0, "to": 1, "weight": 0.8, "type": "explicit"}
  ]
}
```

## Section 7: Migration Plan

### Delete
- All 17 templates in `templates/` (dashboard, goals, cognition, imagine, face, explore, training, spiking, brain, chat, listen, watch, youtube, interact, evolution, experiments, base)

### Create
- `templates/index.html` — single page with Three.js, panels, input bar
- `static/cortex.js` — Three.js brain visualization logic (separate file, not inline)
- `static/cortex.css` — minimal custom styles beyond Tailwind

### Modify
- `brain-server/src/app.rs` — single `GET /` route serving index.html, remove old page routes
- `brain-server/src/routes.rs` — add `api_brain_knowledge_graph` endpoint, remove old page handlers
- Keep ALL existing API endpoints unchanged

### Old Routes
- All old page routes (`/goals`, `/cognition`, `/face`, etc.) redirect to `/`
- All API routes stay unchanged

## Success Criteria

1. Single page loads with a rotating 3D brain showing 10 colored regions
2. Knowledge nodes visible when zoomed into association cortex
3. Typing a question → brain animates → response card with confidence tags
4. Pasting a YouTube URL → learning animation → new concepts appear in brain
5. Slide-out knowledge panel lets you browse topics and click to zoom to concepts
6. Works on desktop Chrome/Firefox (no mobile requirement for v1)
7. Page load under 3 seconds (Three.js from CDN, initial data from 2 API calls)

## Non-Goals

- No mobile optimization (v1 is desktop-focused)
- No authentication or multi-user
- No WebSocket (SSE + polling is sufficient)
- No saved state between browser sessions (brain state is server-side)
- No companion face/chat (replaced by unified input)
- No experiment tracking UI (developer uses API directly)
