"""Frontend is served by FastAPI; no OCR/Gemini involved."""

from fastapi.testclient import TestClient

from app.main import app


def test_frontend_index_loads():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    text = response.text
    assert "NeoStats" in text
    assert "Dashboard" in text
    assert "Upload / Process" in text
    assert "History" in text
    assert "Settings" in text


def test_frontend_assets_and_health_still_work():
    client = TestClient(app)
    css = client.get("/css/styles.css")
    js = client.get("/js/app.js")
    health = client.get("/api/v1/health")
    assert css.status_code == 200
    assert "--burgundy" in css.text
    assert "Manrope" in css.text or "--font" in css.text
    assert js.status_code == 200
    assert health.status_code == 200
    assert health.json()["status"] == "healthy"


def test_frontend_has_a_deployable_runtime_api_configuration():
    client = TestClient(app)
    config = client.get("/js/runtime-config.js")
    api = client.get("/js/api.js")
    assert config.status_code == 200
    assert "NEOSTATS_API_BASE" in config.text
    assert api.status_code == 200
    assert "window.NEOSTATS_API_BASE" in api.text
    assert "localhost" not in api.text
