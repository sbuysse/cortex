//! SSE event bus — broadcast brain events to connected clients.

use serde_json::Value;
use std::sync::Arc;
use tokio::sync::broadcast;

/// Brain event for SSE streaming.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BrainEvent {
    #[serde(rename = "type")]
    pub event_type: String,
    pub time: f64,
    #[serde(flatten)]
    pub data: Value,
}

/// Thread-safe broadcast bus for SSE events.
#[derive(Clone)]
pub struct SseBus {
    sender: broadcast::Sender<BrainEvent>,
}

impl SseBus {
    pub fn new(capacity: usize) -> Self {
        let (sender, _) = broadcast::channel(capacity);
        Self { sender }
    }

    /// Emit an event to all subscribers.
    pub fn emit(&self, event_type: &str, data: Value) {
        let event = BrainEvent {
            event_type: event_type.to_string(),
            time: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs_f64(),
            data,
        };
        // Ignore send errors (no subscribers)
        let _ = self.sender.send(event);
    }

    /// Subscribe to the event stream.
    pub fn subscribe(&self) -> broadcast::Receiver<BrainEvent> {
        self.sender.subscribe()
    }

    /// Number of active subscribers.
    pub fn subscriber_count(&self) -> usize {
        self.sender.receiver_count()
    }
}

impl Default for SseBus {
    fn default() -> Self {
        Self::new(256)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_emit_and_receive() {
        let bus = SseBus::new(16);
        let mut rx = bus.subscribe();
        bus.emit("test", serde_json::json!({"msg": "hello"}));
        let event = rx.recv().await.unwrap();
        assert_eq!(event.event_type, "test");
    }

    #[test]
    fn test_no_subscribers_doesnt_panic() {
        let bus = SseBus::new(16);
        bus.emit("test", serde_json::json!({})); // should not panic
    }
}
