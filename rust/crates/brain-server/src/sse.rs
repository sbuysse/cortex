//! Server-Sent Events for live experiment metrics streaming.

use axum::response::sse::{Event, Sse};
use brain_experiment::runner::MetricsEvent;
use std::convert::Infallible;
use tokio::sync::broadcast;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt;

/// Create an SSE stream filtered by experiment ID.
pub fn experiment_sse_stream(
    rx: broadcast::Receiver<MetricsEvent>,
    experiment_id: i64,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>> {
    let stream = BroadcastStream::new(rx).filter_map(move |result| {
        match result {
            Ok(event) if event.experiment_id == experiment_id => {
                let data = serde_json::to_string(&event).unwrap_or_default();
                Some(Ok(Event::default().data(data).event("metrics")))
            }
            _ => None,
        }
    });

    Sse::new(stream)
}
