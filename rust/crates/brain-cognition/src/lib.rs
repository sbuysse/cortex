//! Brain Cognition — the cognitive architecture in Rust.

pub mod config;
pub mod sse;
pub mod working_memory;
pub mod fast_memory;
pub mod grid_cells;
pub mod memory_db;
pub mod personal_memory;
pub mod concepts;
pub mod dreams;
pub mod search;
pub mod state;
pub mod autonomy;
pub mod personal;
pub mod companion;
pub mod brain_state;

pub use config::BrainConfig;
pub use sse::SseBus;
pub use working_memory::WorkingMemory;
pub use fast_memory::HopfieldMemory;
pub use grid_cells::GridCellEncoder;
pub use memory_db::MemoryDb;
pub use personal_memory::PersonalMemory;
pub use concepts::ConceptCodebook;
pub use state::BrainState;
pub use brain_state::{compose_brain_state, emotion_to_idx, load_emotion_table};
