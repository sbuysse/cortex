//! Application setup — builds the axum Router with all routes.

use axum::routing::{delete, get, post};
use axum::Router;
use brain_db::KnowledgeBase;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tower_http::services::ServeDir;

use crate::routes;
use crate::state::{AppState, BrainViz, InteractState};

fn build_router(state: Arc<AppState>, static_dir: Option<&Path>) -> Router {
    let mut app = Router::new()
        // Single-page UI — all old page routes redirect to /
        .route("/", get(routes::index_page))
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
        // YouTube Brain proxy
        .route("/api/youtube/process", post(routes::api_youtube_process))
        // Listen API proxy
        .route("/api/listen/process", post(routes::api_listen_process))
        // Watch (webcam) + text query proxies
        .route("/api/brain/watch", post(routes::api_brain_watch))
        .route("/api/brain/text_query", post(routes::api_brain_text_query))
        // Online learning + reflection proxies (Phase 4 & 5)
        .route("/api/brain/learn", post(routes::api_brain_proxy_learn))
        .route("/api/brain/learn/train", post(routes::api_brain_learn_train))
        .route("/api/brain/learn/status", get(routes::api_brain_learn_status_native))
        .route("/api/brain/reflect", post(routes::api_brain_reflect))
        .route("/api/brain/reflections", get(routes::api_brain_reflections))
        .route("/api/brain/reflect/auto", post(routes::api_brain_reflect_auto))
        // AGI capabilities (Steps 1-7)
        .route("/api/brain/predict", post(routes::api_brain_predict))
        .route("/api/brain/reason", post(routes::api_brain_reason))
        .route("/api/brain/curiosity", get(routes::api_brain_curiosity))
        .route("/api/brain/curiosity/score", post(routes::api_brain_curiosity_score))
        .route("/api/brain/compose", post(routes::api_brain_compose))
        .route("/api/brain/decompose", post(routes::api_brain_decompose))
        .route("/api/brain/causal/predict", get(routes::api_brain_causal_predict))
        .route("/api/brain/causal/explain", get(routes::api_brain_causal_explain))
        .route("/api/brain/self/assessment", get(routes::api_brain_self_assessment))
        .route("/api/brain/self/confidence", post(routes::api_brain_self_confidence))
        .route("/api/brain/self/progress", get(routes::api_brain_self_progress))
        .route("/api/brain/imagine", post(routes::api_brain_imagine_native))
        .route("/api/brain/autonomy/start", post(routes::api_brain_autonomy_start))
        .route("/api/brain/autonomy/stop", post(routes::api_brain_autonomy_stop))
        .route("/api/brain/autonomy/status", get(routes::api_brain_autonomy_status_native))
        .route("/api/brain/intelligence", get(routes::api_brain_intelligence_native))
        .route("/api/brain/memory/stats", get(routes::api_brain_memory_stats_native))
        .route("/api/brain/memory/recent", get(routes::api_brain_memory_recent_native))
        .route("/api/brain/dialogue", post(routes::api_brain_dialogue))
        // Neuroscience-inspired endpoints
        .route("/api/brain/curiosity/distributional", get(routes::api_brain_curiosity_distributional))
        .route("/api/brain/memory/fast", get(routes::api_brain_fast_memory_native))
        .route("/api/brain/memory/fast/query", post(routes::api_brain_fast_memory_query))
        .route("/api/brain/config", post(routes::api_brain_config_native))
        // Grid Cell System
        .route("/api/brain/grid/map", get(routes::api_brain_grid_map_native))
        .route("/api/brain/grid/navigate", post(routes::api_brain_grid_navigate))
        .route("/api/brain/grid/between", post(routes::api_brain_grid_between))
        .route("/api/brain/grid/episode/{id}", get(routes::api_brain_grid_episode))
        // SSE: Live Brain Activity Stream (native Rust if BrainState available)
        .route("/api/brain/live", get(routes::api_brain_live_native))
        // Feature B: Knowledge Graph (native Rust)
        .route("/api/brain/knowledge", get(routes::api_brain_knowledge_native))
        .route("/api/brain/knowledge/query", post(routes::api_brain_knowledge_query_native))
        .route("/api/brain/knowledge/text", get(routes::api_brain_knowledge_text))
        // Feature D: Text Understanding
        .route("/api/brain/read", post(routes::api_brain_read_native))
        .route("/api/brain/ingest/audioset", post(routes::api_brain_ingest_audioset))
        .route("/api/brain/ingest/wikipedia", post(routes::api_brain_ingest_wikipedia))
        // Feature A: Dreams
        .route("/api/brain/dreams", get(routes::api_brain_dreams_native))
        .route("/api/brain/dream", post(routes::api_brain_dream_native))
        // Option 2: Voice (TTS)
        .route("/api/brain/speak", post(routes::api_brain_speak_native))
        .route("/api/brain/speak/thought", get(routes::api_brain_speak_thought))
        // Option 4: Agentic (web search, research, fetch)
        .route("/api/brain/search", post(routes::api_brain_search_native))
        .route("/api/brain/research", post(routes::api_brain_research))
        .route("/api/brain/fetch", post(routes::api_brain_fetch))
        // Phase 7: YouTube Learning
        .route("/api/brain/youtube_learn", post(routes::api_brain_youtube_learn))
        .route("/api/brain/learn/academic", post(routes::api_brain_learn_academic))
        .route("/api/brain/learn/batch", post(routes::api_brain_learn_batch))
        .route("/api/brain/knowledge/stats", get(routes::api_brain_knowledge_stats))
        .route("/api/brain/knowledge/graph", get(routes::api_brain_knowledge_graph))
        // Phase 2: Episodic Memory
        .route("/api/brain/episodes", get(routes::api_brain_episodes_native))
        .route("/api/brain/remember", post(routes::api_brain_remember))
        .route("/api/brain/predict_next", post(routes::api_brain_predict_next))
        // Phase 3: Concept Hierarchy
        .route("/api/brain/hierarchy", get(routes::api_brain_hierarchy))
        .route("/api/brain/query", post(routes::api_brain_query))
        // Phase 4: Working Memory (native Rust)
        .route("/api/brain/working_memory", get(routes::api_brain_working_memory_native))
        // Phase 5: Prototypes (native Rust)
        .route("/api/brain/prototypes", get(routes::api_brain_prototypes_native))
        .route("/api/brain/prototypes/add", post(routes::api_brain_prototypes_add))
        .route("/api/brain/consolidate", post(routes::api_brain_consolidate_native))
        // Phase 6: Goal-Directed Planning
        .route("/api/brain/plan", post(routes::api_brain_plan_native))
        // Phase 8: Language Generation
        .route("/api/brain/thoughts", get(routes::api_brain_thoughts_native))
        .route("/api/brain/think", post(routes::api_brain_think_native))
        .route("/api/brain/dialogue/grounded", post(routes::api_brain_dialogue_grounded_native))
        // Companion API
        .route("/api/companion/greeting", get(routes::api_companion_greeting))
        .route("/api/companion/safety", get(routes::api_companion_safety))
        .route("/api/companion/personal", get(routes::api_companion_personal))
        // Spiking brain status
        .route("/api/brain/spiking/status", get(routes::api_brain_spiking_status))
        // Brain Voice API (catch-all proxy)
        .route("/api/brain/{endpoint}", post(routes::api_brain_proxy))
        .route("/api/brain/chat/{session_id}", delete(routes::api_brain_chat_delete))
        // Brain visualization API
        .route("/api/brain/state", get(routes::api_brain_state))
        .route("/api/brain/clip", get(routes::api_brain_clip))
        // JSON API (existing)
        .route("/api/status", get(routes::api_status))
        .route("/api/timeline", get(routes::api_timeline))
        .route("/api/experiment/{id}", get(routes::api_experiment))
        .route("/api/logs", get(routes::api_logs))
        .route("/api/decisions", get(routes::api_decisions))
        .route("/api/history", get(routes::api_history))
        // JSON API (new)
        .route("/api/experiments", get(routes::api_experiments_list))
        .route("/api/mutations", get(routes::api_mutations_list))
        .route("/api/mutations/stats", get(routes::api_mutations_stats))
        // Interactive retrieval API
        .route("/api/interact/retrieve", get(routes::api_interact_retrieve))
        .route("/api/interact/random", get(routes::api_interact_random))
        .route("/api/interact/labels", get(routes::api_interact_labels))
        .route("/api/interact/search", get(routes::api_interact_search))
        // SSE
        .route(
            "/api/live/experiment/{id}",
            get(routes::api_live_experiment),
        )
        .with_state(state);

    if let Some(static_dir) = static_dir {
        app = app.nest_service("/static", ServeDir::new(static_dir));
    }

    app
}

/// Build router AND return shared state (for cortex access).
pub fn build_app_with_state(
    db_path: &Path,
    templates_dir: &Path,
    static_dir: Option<&Path>,
    project_root: PathBuf,
    output_dir: PathBuf,
    interact: Option<Arc<InteractState>>,
    brain_viz: Option<Arc<BrainViz>>,
) -> (Router, Arc<AppState>) {
    let db = KnowledgeBase::new(db_path).expect("Failed to open database");
    let state = Arc::new(AppState::new(db, templates_dir, project_root, output_dir, interact, brain_viz));
    let router = build_router(state.clone(), static_dir);
    (router, state)
}

/// Start the server on the given address.
pub async fn serve(app: Router, addr: &str) -> std::io::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("Brain server listening on {addr}");
    axum::serve(listener, app).await?;
    Ok(())
}
