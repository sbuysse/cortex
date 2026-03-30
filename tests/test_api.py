"""API endpoint smoke tests — verify all endpoints return valid HTTP status codes."""
import pytest
import requests
import time

BASE = "http://localhost:8099"
TIMEOUT = 10


def is_service_running():
    try:
        r = requests.get(f"{BASE}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not is_service_running(),
    reason="Brain service not running on localhost:8099"
)


class TestGetEndpoints:
    """All GET endpoints should return 200."""

    @pytest.mark.parametrize("path", [
        "/health",
        "/api/brain/working_memory",
        "/api/brain/prototypes",
        "/api/brain/episodes",
        "/api/brain/hierarchy",
        "/api/brain/knowledge",
        "/api/brain/knowledge/text",
        "/api/brain/dreams",
        "/api/brain/memory/fast",
        "/api/brain/grid/map",
        "/api/brain/autonomy/status",
        "/api/brain/learn/status",
        "/api/brain/memory/stats",
        "/api/brain/memory/recent",
        "/api/brain/intelligence",
        "/api/brain/thoughts",
        "/api/brain/curiosity",
        "/api/brain/curiosity/distributional",
        "/api/brain/self/assessment",
        "/api/brain/self/progress",
    ])
    def test_get_endpoint(self, path):
        r = requests.get(f"{BASE}{path}", timeout=TIMEOUT)
        assert r.status_code == 200, f"{path} returned {r.status_code}: {r.text[:200]}"
        data = r.json()
        assert isinstance(data, dict), f"{path} should return dict, got {type(data)}"


class TestPostEndpoints:
    """POST endpoints with valid data should return 200 or 422 (validation)."""

    def test_knowledge_query(self):
        r = requests.post(f"{BASE}/api/brain/knowledge/query",
                          json={"start": "thunder", "max_hops": 2}, timeout=TIMEOUT)
        assert r.status_code == 200

    def test_grid_navigate(self):
        r = requests.post(f"{BASE}/api/brain/grid/navigate",
                          json={"query": "thunder", "top_k": 5}, timeout=TIMEOUT)
        assert r.status_code == 200

    def test_grid_between(self):
        r = requests.post(f"{BASE}/api/brain/grid/between",
                          json={"concept_a": "thunder", "concept_b": "rain"}, timeout=TIMEOUT)
        assert r.status_code == 200

    def test_read_text(self):
        r = requests.post(f"{BASE}/api/brain/read",
                          json={"text": "Test text for reading", "source": "test"}, timeout=TIMEOUT)
        assert r.status_code == 200

    def test_config(self):
        r = requests.post(f"{BASE}/api/brain/config",
                          json={"sparse_k": 0}, timeout=TIMEOUT)
        assert r.status_code == 200

    def test_remember(self):
        r = requests.post(f"{BASE}/api/brain/remember",
                          json={"sequence": ["thunder", "rain"]}, timeout=30)
        assert r.status_code == 200

    def test_search(self):
        r = requests.post(f"{BASE}/api/brain/search",
                          json={"query": "thunder sound"}, timeout=TIMEOUT)
        assert r.status_code == 200


class TestHealthCheck:
    """Health endpoint should return component status."""

    def test_health_components(self):
        r = requests.get(f"{BASE}/health", timeout=TIMEOUT)
        data = r.json()
        assert "components" in data
        assert "mlp" in data["components"]
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] > 0

    def test_health_memory(self):
        r = requests.get(f"{BASE}/health", timeout=TIMEOUT)
        data = r.json()
        assert "memory" in data
        assert "prototypes" in data["memory"]


class TestSSE:
    """SSE endpoint should stream events."""

    def test_sse_connection(self):
        r = requests.get(f"{BASE}/api/brain/live", stream=True, timeout=5)
        assert r.status_code == 200
        assert "text/event-stream" in r.headers.get("content-type", "")
        # Read first event
        for line in r.iter_lines(decode_unicode=True):
            if line.startswith("data:"):
                import json
                event = json.loads(line[5:])
                assert "type" in event
                break
        r.close()
