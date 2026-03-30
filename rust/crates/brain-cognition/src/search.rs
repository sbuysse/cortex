//! Search — semantic text search and web search capabilities.

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct SearchResult {
    pub label: String,
    pub similarity: f32,
    pub source: String,
}

/// Semantic search: text query → nearest category labels.
pub fn text_semantic_search(
    query: &str,
    text_encoder: &brain_inference::TextEncoder,
    top_k: usize,
) -> Result<Vec<SearchResult>, Box<dyn std::error::Error>> {
    let results = text_encoder.semantic_search(query, top_k)?;
    Ok(results.into_iter().map(|(label, sim)| SearchResult {
        label,
        similarity: sim,
        source: "vggsound".into(),
    }).collect())
}

/// Web search via DuckDuckGo HTML API.
pub async fn web_search(query: &str, max_results: usize) -> Vec<WebSearchResult> {
    let encoded = urlencoding::encode(query);
    let url = format!("https://html.duckduckgo.com/html/?q={encoded}");

    let client = reqwest::Client::new();
    let resp = match client.get(&url)
        .header("User-Agent", "BrainProject/1.0")
        .timeout(std::time::Duration::from_secs(15))
        .send().await {
        Ok(r) => r,
        Err(_) => return Vec::new(),
    };

    let html = match resp.text().await {
        Ok(t) => t,
        Err(_) => return Vec::new(),
    };

    // Parse results from DuckDuckGo HTML
    let mut results = Vec::new();
    let re = regex_lite::Regex::new(r#"class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>"#).unwrap();
    for cap in re.captures_iter(&html) {
        if results.len() >= max_results { break; }
        let mut link = cap.get(1).map(|m| m.as_str().to_string()).unwrap_or_default();
        let title = cap.get(2).map(|m| {
            // Strip HTML tags
            regex_lite::Regex::new(r"<[^>]+>").unwrap().replace_all(m.as_str(), "").trim().to_string()
        }).unwrap_or_default();

        // DDG wraps URLs
        if link.contains("uddg=") {
            if let Some(decoded) = link.split("uddg=").nth(1) {
                link = urlencoding::decode(decoded.split('&').next().unwrap_or("")).unwrap_or_default().to_string();
            }
        }

        if !title.is_empty() && !link.is_empty() {
            results.push(WebSearchResult { title, url: link });
        }
    }

    results
}

#[derive(Debug, Clone, Serialize)]
pub struct WebSearchResult {
    pub title: String,
    pub url: String,
}
