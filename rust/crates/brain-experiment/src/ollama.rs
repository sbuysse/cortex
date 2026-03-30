//! Ollama HTTP API client for mutation generation.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct OllamaClient {
    base_url: String,
    model: String,
    client: reqwest::Client,
}

#[derive(Debug, Serialize)]
struct GenerateRequest<'a> {
    model: &'a str,
    prompt: &'a str,
    stream: bool,
    options: GenerateOptions,
}

#[derive(Debug, Serialize)]
struct GenerateOptions {
    temperature: f32,
    num_predict: u32,
    top_p: f32,
}

#[derive(Debug, Deserialize)]
struct GenerateResponse {
    response: String,
}

#[derive(Debug, thiserror::Error)]
pub enum OllamaError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("Ollama not reachable")]
    NotReachable,
    #[error("Empty response from LLM")]
    EmptyResponse,
}

impl OllamaClient {
    pub fn new(base_url: &str, model: &str) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .expect("Failed to build HTTP client");

        Self {
            base_url: base_url.to_string(),
            model: model.to_string(),
            client,
        }
    }

    pub fn default() -> Self {
        Self::new("http://localhost:11434", "qwen2.5-coder:14b")
    }

    /// Check if Ollama is reachable.
    pub async fn is_available(&self) -> bool {
        match self.client
            .get(format!("{}/api/tags", self.base_url))
            .timeout(std::time::Duration::from_secs(5))
            .send()
            .await
        {
            Ok(resp) => resp.status().is_success(),
            Err(e) => {
                tracing::debug!(error = %e, url = %self.base_url, "Ollama availability check failed");
                false
            }
        }
    }

    /// Generate text from Ollama.
    pub async fn generate(
        &self,
        prompt: &str,
        temperature: f32,
    ) -> Result<String, OllamaError> {
        let req = GenerateRequest {
            model: &self.model,
            prompt,
            stream: false,
            options: GenerateOptions {
                temperature,
                num_predict: 1024,
                top_p: 0.9,
            },
        };

        let resp = self
            .client
            .post(format!("{}/api/generate", self.base_url))
            .json(&req)
            .send()
            .await?;

        let body: GenerateResponse = resp.json().await?;
        if body.response.trim().is_empty() {
            return Err(OllamaError::EmptyResponse);
        }
        Ok(body.response)
    }
}
