//! SQLite-backed experiment log and decision journal.
//!
//! Direct port of the Python KnowledgeBase using the same schema.

pub mod models;
pub mod knowledge_base;

pub use knowledge_base::KnowledgeBase;
pub use models::*;
