"""
API FastAPI pour la prédiction de Qualité de Service (QoS) 5G et la détection d'anomalies.
Endpoints principaux :
- POST /predict : Prédiction de la QoS (bon, moyen, dégradé) et estimation du score continu
- GET /anomalies : Récupération des anomalies réseau détectées par Isolation Forest
- GET /health : Vérification de la disponibilité du service
- GET /metrics : Métriques de performance des modèles (RMSE, R², F1, recall)
"""

import os
import sys
import json
import logging
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager
from datetime import datetime

# Garantit que la racine est dans sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from src.preprocessing import QoSPreprocessor, INV_QUALITY_LABEL_MAP

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("qos_5g_api")

MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "data", "qos_5g.csv")

# Variables globales pour les modèles en mémoire
models: Dict[str, Any] = {}


def load_artifacts():
    """Charge les modèles entraînés et le préprocesseur."""
    try:
        models["preprocessor"] = joblib.load(os.path.join(MODELS_DIR, "preprocessor.joblib"))
        models["xgb_clf"] = joblib.load(os.path.join(MODELS_DIR, "xgboost_clf.joblib"))
        models["rf_clf"] = joblib.load(os.path.join(MODELS_DIR, "random_forest_clf.joblib"))
        models["xgb_reg"] = joblib.load(os.path.join(MODELS_DIR, "xgboost_reg.joblib"))
        models["linear_reg"] = joblib.load(os.path.join(MODELS_DIR, "baseline_linear_reg.joblib"))
        models["isolation_forest"] = joblib.load(os.path.join(MODELS_DIR, "isolation_forest.joblib"))

        summary_file = os.path.join(MODELS_DIR, "evaluation_summary.json")
        if os.path.exists(summary_file):
            with open(summary_file, "r", encoding="utf-8") as f:
                models["summary"] = json.load(f)
        else:
            models["summary"] = {}

        logger.info("Tous les artefacts ML ont été chargés avec succès depuis %s.", MODELS_DIR)
    except Exception as e:
        logger.error("Erreur lors du chargement des artefacts : %s", e)

# Chargement direct au démarrage du module
load_artifacts()


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Gestionnaire de cycle de vie : chargement des artefacts au démarrage."""
    if not models or "preprocessor" not in models:
        load_artifacts()
    yield


app = FastAPI(
    title="API de Prédiction QoS Réseau 5G",
    description="Microservice FastAPI pour prédire la Qualité de Service (QoS) et détecter les anomalies de cellules 5G (eMBB, URLLC, mMTC, HC).",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configuration CORS pour intégration web et dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================
# Schémas Pydantic
# ==============================================================
class CellKPIInput(BaseModel):
    cell_id: str = Field(default="CELL_001", description="Identifiant unique de la cellule 5G (ex: CELL_001)")
    cell_type: str = Field(default="macro", description="Type d'antenne : macro, micro, pico")
    slice_type: str = Field(default="eMBB", description="Type de tranche de réseau : eMBB, URLLC, mMTC, HC")
    throughput_mbps: float = Field(default=750.5, ge=0.0, description="Débit moyen mesuré en Mbps")
    latency_ms: float = Field(default=8.5, ge=0.0, description="Latence aller-retour mesurée en millisecondes")
    packet_loss_pct: float = Field(default=0.15, ge=0.0, le=100.0, description="Pourcentage de paquets perdus (0 à 100%)")
    handover_count: int = Field(default=2, ge=0, description="Nombre de basculements de cellule observés")
    rsrp_dbm: float = Field(default=-95.0, description="Puissance du signal radio reçu en dBm")
    rsrq_db: float = Field(default=-11.0, description="Qualité du signal radio reçu en dB")
    prb_utilization_pct: Optional[float] = Field(default=65.0, ge=0.0, le=100.0, description="Taux d'utilisation des blocs de ressources radio PRB")
    active_users: Optional[int] = Field(default=45, ge=0, description="Nombre d'utilisateurs connectés actifs")
    sla_compliant: Optional[int] = Field(default=1, ge=0, le=1, description="Indicateur de conformité SLA (1=oui, 0=non)")
    timestamp: Optional[str] = Field(default=None, description="Horodatage ISO facultatif")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "cell_id": "CELL_007",
                "cell_type": "macro",
                "slice_type": "eMBB",
                "throughput_mbps": 850.2,
                "latency_ms": 7.8,
                "packet_loss_pct": 0.12,
                "handover_count": 2,
                "rsrp_dbm": -92.5,
                "rsrq_db": -9.8,
                "prb_utilization_pct": 68.0,
                "active_users": 42,
                "sla_compliant": 1
            }
        }
    )


class PredictionResponse(BaseModel):
    predicted_quality: str = Field(..., description="Classe prédite : bon, moyen, dégradé")
    quality_code: int = Field(..., description="Code numérique de la classe (0=bon, 1=moyen, 2=dégradé)")
    confidence: float = Field(..., description="Probabilité de la classe prédite")
    class_probabilities: Dict[str, float] = Field(..., description="Probabilités pour chaque classe")
    estimated_qos_score: float = Field(..., description="Score global continu de QoS estimé (0 à 100)")
    is_anomaly: bool = Field(..., description="Indique si cette mesure est classée comme anomalie par Isolation Forest")
    anomaly_score: float = Field(..., description="Score d'anomalie de l'Isolation Forest (plus bas = plus anormal)")
    model_used: str = Field(default="XGBoost Classifier")
    timestamp: str


class AnomalyItem(BaseModel):
    timestamp: str
    cell_id: str
    cell_type: str
    slice_type: str
    throughput_mbps: float
    latency_ms: float
    packet_loss_pct: float
    rsrp_dbm: float
    rsrq_db: float
    prb_utilization_pct: Optional[float] = None
    anomaly_score: float
    reasons: List[str]


# ==============================================================
# Routes FastAPI
# ==============================================================
@app.get("/health", tags=["Système"])
def health_check():
    """Vérifie l'état de santé du microservice et des modèles."""
    models_ready = all(k in models for k in ["preprocessor", "xgb_clf", "isolation_forest"])
    return {
        "status": "ok" if models_ready else "degraded",
        "service": "5G-QoS-Prediction-API",
        "models_loaded": list(models.keys()),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/metrics", tags=["Métriques ML"])
def get_model_metrics():
    """Renvoie les métriques détaillées d'évaluation de tous les modèles entraînés."""
    if "summary" not in models or not models["summary"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Les métriques d'évaluation ne sont pas encore disponibles."
        )
    return models["summary"]


@app.post("/predict", response_model=PredictionResponse, tags=["Prédiction"])
def predict_qos(payload: CellKPIInput):
    """
    Prédit la qualité de service d'une cellule 5G à partir de ses KPIs.
    Renvoie la classe de qualité ('bon', 'moyen', 'dégradé'), les probabilités,
    le score continu et la détection d'anomalie.
    """
    if "preprocessor" not in models or "xgb_clf" not in models:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Les modèles ML ne sont pas encore prêts."
        )

    try:
        prep: QoSPreprocessor = models["preprocessor"]
        clf = models["xgb_clf"]
        reg = models.get("xgb_reg", models.get("linear_reg"))
        iso = models.get("isolation_forest")

        instance_data = payload.model_dump()
        if not instance_data.get("timestamp"):
            instance_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Transformation par le pipeline
        X_scaled = prep.prepare_single_instance(instance_data)

        # Prédiction de classe & probabilités
        pred_class_idx = int(clf.predict(X_scaled)[0])
        pred_label = INV_QUALITY_LABEL_MAP.get(pred_class_idx, "inconnu")

        probabilities = clf.predict_proba(X_scaled)[0]
        prob_dict = {
            INV_QUALITY_LABEL_MAP.get(i, f"class_{i}"): round(float(p), 4)
            for i, p in enumerate(probabilities)
        }
        confidence = float(np.max(probabilities))

        # Score continu
        qos_score = float(reg.predict(X_scaled)[0]) if reg else 50.0
        qos_score = max(0.0, min(100.0, qos_score))

        # Détection d'anomalie
        is_anomaly = False
        anomaly_score = 0.0
        if iso:
            anomaly_pred = iso.predict(X_scaled)[0]  # -1 = anomalie, 1 = normal
            is_anomaly = bool(anomaly_pred == -1)
            anomaly_score = float(iso.decision_function(X_scaled)[0])

        return PredictionResponse(
            predicted_quality=pred_label,
            quality_code=pred_class_idx,
            confidence=round(confidence, 4),
            class_probabilities=prob_dict,
            estimated_qos_score=round(qos_score, 2),
            is_anomaly=is_anomaly,
            anomaly_score=round(anomaly_score, 4),
            model_used="XGBoost Classifier + Isolation Forest",
            timestamp=instance_data["timestamp"]
        )

    except Exception as e:
        logger.error("Erreur durant la prédiction : %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur interne lors de la prédiction : {str(e)}")


@app.get("/anomalies", response_model=List[AnomalyItem], tags=["Détection d'Anomalies"])
def get_anomalies(
    limit: int = Query(default=20, ge=1, le=100, description="Nombre maximum d'anomalies à retourner"),
    slice_filter: Optional[str] = Query(default=None, description="Filtrer par tranche : eMBB, URLLC, mMTC, HC"),
    cell_filter: Optional[str] = Query(default=None, description="Filtrer par ID cellule (ex: CELL_007)")
):
    """
    Récupère la liste des anomalies détectées par l'Isolation Forest dans le dataset 5G.
    Fournit les motifs potentiels (perte de paquet excessive, dégradation radio, congestion PRB).
    """
    if not os.path.exists(DATA_PATH) or "preprocessor" not in models or "isolation_forest" not in models:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Données ou modèle d'anomalie non chargés."
        )

    try:
        prep: QoSPreprocessor = models["preprocessor"]
        iso = models["isolation_forest"]

        df = pd.read_csv(DATA_PATH)
        
        # Filtres optionnels
        if slice_filter:
            df = df[df["slice_type"].str.upper() == slice_filter.strip().upper()]
        if cell_filter:
            df = df[df["cell_id"].str.upper() == cell_filter.strip().upper()]

        if df.empty:
            return []

        # Échantillonnage ou analyse des 500 dernières lignes pour rapidité
        recent_df = df.tail(min(1500, len(df))).copy().reset_index(drop=True)
        X_scaled, _, _ = prep.transform(recent_df)
        
        preds = iso.predict(X_scaled)
        scores = iso.decision_function(X_scaled)

        anomaly_indices = np.where(preds == -1)[0]
        
        results: List[AnomalyItem] = []
        for idx in anomaly_indices:
            row = recent_df.iloc[idx]
            reasons = []

            # Analyse des raisons probables de l'anomalie
            if row.get("packet_loss_pct", 0) > 2.0:
                reasons.append(f"Perte de paquets anormale ({row['packet_loss_pct']}%)")
            if row.get("latency_ms", 0) > 30.0:
                reasons.append(f"Latence très élevée ({row['latency_ms']} ms)")
            if row.get("prb_utilization_pct", 0) > 95.0:
                reasons.append(f"Saturation critique des ressources PRB ({row['prb_utilization_pct']}%)")
            if row.get("rsrp_dbm", 0) < -115.0:
                reasons.append(f"Signal radio critique ({row['rsrp_dbm']} dBm)")
            if row.get("handover_count", 0) >= 8:
                reasons.append(f"Instabilité de mobilité / ping-pong handover ({row['handover_count']} hnd)")
            if not reasons:
                reasons.append("Déviation multivariée atypique détectée par Isolation Forest")

            item = AnomalyItem(
                timestamp=str(row["timestamp"]),
                cell_id=str(row["cell_id"]),
                cell_type=str(row["cell_type"]),
                slice_type=str(row["slice_type"]),
                throughput_mbps=float(row["throughput_mbps"]),
                latency_ms=float(row["latency_ms"]),
                packet_loss_pct=float(row["packet_loss_pct"]),
                rsrp_dbm=float(row["rsrp_dbm"]),
                rsrq_db=float(row["rsrq_db"]),
                prb_utilization_pct=float(row["prb_utilization_pct"]) if "prb_utilization_pct" in row else None,
                anomaly_score=round(float(scores[idx]), 4),
                reasons=reasons
            )
            results.append(item)
            if len(results) >= limit:
                break

        return results

    except Exception as e:
        logger.error("Erreur lors de la récupération des anomalies : %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur d'extraction des anomalies : {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
