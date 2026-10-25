"""
Tests d'intégration pour l'API FastAPI QoS 5G (api/main.py).
Utilise httpx.TestClient (ou le TestClient de Starlette) pour tester :
- GET /health     : disponibilité du microservice et des modèles
- GET /metrics    : rapport d'évaluation des modèles ML
- POST /predict   : prédiction QoS sur une cellule simulée
- GET /anomalies  : liste des anomalies Isolation Forest avec filtres optionnels
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Payload valide de référence (cellule eMBB en bonne santé)
# ---------------------------------------------------------------------------
VALID_PAYLOAD = {
    "cell_id": "CELL_007",
    "cell_type": "macro",
    "slice_type": "eMBB",
    "throughput_mbps": 850.0,
    "latency_ms": 7.8,
    "packet_loss_pct": 0.12,
    "handover_count": 2,
    "rsrp_dbm": -92.5,
    "rsrq_db": -9.8,
    "prb_utilization_pct": 68.0,
    "active_users": 42,
    "sla_compliant": 1,
}

DEGRADED_PAYLOAD = {
    "cell_id": "CELL_003",
    "cell_type": "macro",
    "slice_type": "eMBB",
    "throughput_mbps": 200.0,
    "latency_ms": 40.0,
    "packet_loss_pct": 4.0,
    "handover_count": 5,
    "rsrp_dbm": -118.0,
    "rsrq_db": -18.5,
    "prb_utilization_pct": 98.0,
    "active_users": 120,
    "sla_compliant": 0,
}

URLLC_PAYLOAD = {
    "cell_id": "CELL_010",
    "cell_type": "micro",
    "slice_type": "URLLC",
    "throughput_mbps": 200.0,
    "latency_ms": 1.5,
    "packet_loss_pct": 0.01,
    "handover_count": 1,
    "rsrp_dbm": -88.0,
    "rsrq_db": -8.0,
    "prb_utilization_pct": 48.0,
    "active_users": 10,
    "sla_compliant": 1,
}


# ===========================================================================
# Tests GET /health
# ===========================================================================

def test_health_returns_200():
    """Le endpoint /health doit répondre 200."""
    response = client.get("/health")
    assert response.status_code == 200


def test_health_payload_structure():
    """La réponse /health contient les champs attendus."""
    data = client.get("/health").json()
    assert "status" in data
    assert "service" in data
    assert "models_loaded" in data
    assert "timestamp" in data


def test_health_models_loaded():
    """Le service doit indiquer au moins preprocessor et xgb_clf chargés."""
    data = client.get("/health").json()
    assert "preprocessor" in data["models_loaded"]
    assert "xgb_clf" in data["models_loaded"]
    assert data["status"] == "ok"


# ===========================================================================
# Tests GET /metrics
# ===========================================================================

def test_metrics_returns_200():
    """Le endpoint /metrics doit répondre 200 si les modèles sont entraînés."""
    response = client.get("/metrics")
    assert response.status_code == 200


def test_metrics_payload_keys():
    """Le rapport de métriques contient les catégories attendues."""
    data = client.get("/metrics").json()
    assert "classification_metrics" in data
    assert "regression_metrics" in data


def test_metrics_classification_models_present():
    """Au moins XGBoost et Random Forest doivent figurer dans les métriques."""
    data = client.get("/metrics").json()
    cls = data["classification_metrics"]
    assert "xgboost" in cls
    assert "random_forest" in cls


def test_metrics_regression_r2_positive():
    """Le R2 de XGBoost en régression doit être positif (modèle utile)."""
    data = client.get("/metrics").json()
    r2 = data["regression_metrics"]["xgboost"]["r2"]
    assert r2 > 0.0, f"R2 XGBoost attendu > 0 mais obtenu : {r2}"


# ===========================================================================
# Tests POST /predict
# ===========================================================================

def test_predict_valid_payload():
    """Une requête valide doit retourner 200."""
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200


def test_predict_response_structure():
    """La réponse de prédiction doit contenir tous les champs du modèle Pydantic."""
    data = client.post("/predict", json=VALID_PAYLOAD).json()
    expected_keys = [
        "predicted_quality",
        "quality_code",
        "confidence",
        "class_probabilities",
        "estimated_qos_score",
        "is_anomaly",
        "anomaly_score",
        "model_used",
        "timestamp",
    ]
    for key in expected_keys:
        assert key in data, f"Cle manquante dans la reponse : '{key}'"


def test_predict_quality_values():
    """La classe predite doit etre l'une des 3 valeurs valides."""
    data = client.post("/predict", json=VALID_PAYLOAD).json()
    assert data["predicted_quality"] in {"bon", "moyen", "degradé"}


def test_predict_confidence_range():
    """La confiance doit etre comprise entre 0 et 1."""
    data = client.post("/predict", json=VALID_PAYLOAD).json()
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_qos_score_range():
    """Le score continu de QoS doit etre dans [0, 100]."""
    data = client.post("/predict", json=VALID_PAYLOAD).json()
    assert 0.0 <= data["estimated_qos_score"] <= 100.0


def test_predict_class_probabilities_sum():
    """Les probabilites doivent sommer a ~1.0."""
    data = client.post("/predict", json=VALID_PAYLOAD).json()
    total_prob = sum(data["class_probabilities"].values())
    assert abs(total_prob - 1.0) < 1e-3, f"Somme des probabilites = {total_prob}"


def test_predict_quality_code_coherence():
    """Le quality_code doit correspondre a la classe textuelle predite."""
    label_to_code = {"bon": 0, "moyen": 1}
    data = client.post("/predict", json=VALID_PAYLOAD).json()
    quality = data["predicted_quality"]
    code = data["quality_code"]
    # code must be 0, 1 or 2
    assert code in (0, 1, 2)


def test_predict_degraded_scenario():
    """Une cellule fortement degradee devrait retourner la classe degradee."""
    data = client.post("/predict", json=DEGRADED_PAYLOAD).json()
    # degraded scenario: quality should not be "bon"
    assert data["predicted_quality"] in {"moyen", "dégradé"}, (
        f"Cellule tres degradee attendue en moyen/degrade, obtenu : {data['predicted_quality']}"
    )


def test_predict_urllc_slice():
    """Une cellule URLLC conforme aux SLAs doit retourner bon ou moyen."""
    data = client.post("/predict", json=URLLC_PAYLOAD).json()
    assert data["predicted_quality"] in {"bon", "moyen"}, (
        f"Cellule URLLC bien configuree attendue en bon/moyen, obtenu : {data['predicted_quality']}"
    )


def test_predict_missing_optional_fields():
    """Les champs optionnels absents ne doivent pas planter l'API."""
    minimal = {
        "cell_id": "CELL_001",
        "cell_type": "micro",
        "slice_type": "mMTC",
        "throughput_mbps": 35.0,
        "latency_ms": 20.0,
        "packet_loss_pct": 0.3,
        "handover_count": 2,
        "rsrp_dbm": -95.0,
        "rsrq_db": -12.0,
    }
    response = client.post("/predict", json=minimal)
    assert response.status_code == 200


def test_predict_invalid_packet_loss():
    """Un packet_loss_pct > 100% doit declencher une erreur de validation."""
    bad_payload = {**VALID_PAYLOAD, "packet_loss_pct": 150.0}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_negative_throughput():
    """Un debit negatif doit declencher une erreur de validation Pydantic."""
    bad_payload = {**VALID_PAYLOAD, "throughput_mbps": -100.0}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_is_anomaly_type():
    """Le champ is_anomaly doit etre un booleen."""
    data = client.post("/predict", json=VALID_PAYLOAD).json()
    assert isinstance(data["is_anomaly"], bool)


# ===========================================================================
# Tests GET /anomalies
# ===========================================================================

def test_anomalies_returns_200():
    """Le endpoint /anomalies doit retourner 200."""
    response = client.get("/anomalies")
    assert response.status_code == 200


def test_anomalies_returns_list():
    """La reponse doit etre une liste (meme vide)."""
    data = client.get("/anomalies").json()
    assert isinstance(data, list)


def test_anomalies_limit_respected():
    """Le parametre limit doit etre respecte."""
    limit = 5
    data = client.get(f"/anomalies?limit={limit}").json()
    assert len(data) <= limit


def test_anomalies_item_structure():
    """Chaque anomalie retournee doit contenir les champs requis."""
    data = client.get("/anomalies?limit=3").json()
    required_keys = [
        "timestamp", "cell_id", "cell_type", "slice_type",
        "throughput_mbps", "latency_ms", "packet_loss_pct",
        "rsrp_dbm", "rsrq_db", "anomaly_score", "reasons"
    ]
    for item in data:
        for key in required_keys:
            assert key in item, f"Cle '{key}' manquante dans l'anomalie : {item}"


def test_anomalies_reasons_is_list():
    """Le champ reasons doit etre une liste de chaines."""
    data = client.get("/anomalies?limit=5").json()
    for item in data:
        assert isinstance(item["reasons"], list)
        for reason in item["reasons"]:
            assert isinstance(reason, str)


def test_anomalies_slice_filter():
    """Le filtre par slice doit limiter les resultats a cette tranche."""
    data = client.get("/anomalies?slice_filter=eMBB&limit=20").json()
    for item in data:
        assert item["slice_type"].upper() == "EMBB", (
            f"Slice inattendu : {item['slice_type']}"
        )


def test_anomalies_cell_filter():
    """Le filtre par cellule doit limiter les resultats a cette cellule."""
    target_cell = "CELL_001"
    data = client.get(f"/anomalies?cell_filter={target_cell}&limit=10").json()
    for item in data:
        assert item["cell_id"].upper() == target_cell.upper(), (
            f"Cellule inattendue : {item['cell_id']}"
        )


def test_anomalies_limit_invalid():
    """Une limite invalide (ex: 0) doit retourner une erreur 422."""
    response = client.get("/anomalies?limit=0")
    assert response.status_code == 422


def test_anomalies_limit_max():
    """Une limite de 100 (maximum autorise) ne doit pas planter."""
    response = client.get("/anomalies?limit=100")
    assert response.status_code == 200
