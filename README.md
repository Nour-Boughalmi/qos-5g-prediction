# 📶 5G QoS Intelligence Suite

Plateforme prédictive complète de **supervision et d'anticipation de la Qualité de Service (QoS) pour les réseaux 5G**, combinant un pipeline ML avancé, une API REST et un dashboard interactif.

---

## 🗂️ Structure du Projet

```
QoS/
├── api/
│   ├── __init__.py
│   └── main.py              # Microservice FastAPI (prédiction + anomalies)
├── app/
│   ├── __init__.py
│   └── dashboard.py         # Dashboard Streamlit interactif
├── data/
│   ├── generate_dataset.py  # Générateur de données KPI 5G simulées
│   └── qos_5g.csv           # Dataset 5G (4 jours, 20 cellules, 4 slices)
├── models/                  # Artefacts entraînés (.joblib) + métriques (.json)
├── src/
│   ├── __init__.py
│   ├── preprocessing.py     # Pipeline de prétraitement QoS
│   ├── train.py             # Entraînement et évaluation des modèles
│   └── evaluate.py          # Métriques : RMSE, R², F1, Recall, Matrice
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py  # Tests unitaires preprocessing
│   └── test_api.py            # Tests d'intégration FastAPI
└── requirements.txt
```

---

## 🚀 Installation et Démarrage Rapide

### 1. Prérequis
- Python 3.10+
- pip ou conda

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Générer le dataset (si nécessaire)

```bash
python data/generate_dataset.py
```

Génère `data/qos_5g.csv` : **5 760 mesures** sur 4 jours, 20 cellules (macro/micro/pico), 4 types de tranches (eMBB, URLLC, mMTC, HC).

### 4. Entraîner les modèles

```bash
python src/train.py
```

Entraîne 7 modèles ML et sauvegarde les artefacts dans `models/` :
- `preprocessor.joblib`
- `baseline_linear_reg.joblib`
- `baseline_logistic_clf.joblib`
- `random_forest_clf.joblib`
- `random_forest_reg.joblib`
- `xgboost_clf.joblib`
- `xgboost_reg.joblib`
- `isolation_forest.joblib`
- `evaluation_summary.json`

### 5. Lancer l'API FastAPI

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Documentation interactive disponible sur : **http://localhost:8000/docs**

### 6. Lancer le Dashboard Streamlit

```bash
streamlit run app/dashboard.py
```

Dashboard disponible sur : **http://localhost:8501**

---

## 🧠 Modèles ML Intégrés

| Modèle | Type | Cible |
|--------|------|-------|
| Régression Linéaire (Baseline) | Régression | Score QoS continu (0-100) |
| Régression Logistique (Baseline) | Classification | Bon / Moyen / Dégradé |
| Random Forest Classifier | Classification | Bon / Moyen / Dégradé |
| Random Forest Regressor | Régression | Score QoS continu (0-100) |
| **XGBoost Classifier** | Classification | Bon / Moyen / Dégradé |
| **XGBoost Regressor** | Régression | Score QoS continu (0-100) |
| **Isolation Forest** | Anomalie (non-supervisé) | Détection de cellules défaillantes |

---

## 🌐 Endpoints API

### `GET /health`
Vérification de l'état du service.

```json
{
  "status": "ok",
  "service": "5G-QoS-Prediction-API",
  "models_loaded": ["preprocessor", "xgb_clf", ...],
  "timestamp": "2024-01-04T12:00:00"
}
```

### `GET /metrics`
Rapport d'évaluation complet de tous les modèles (RMSE, R², F1, Recall, Feature Importances).

### `POST /predict`
Prédiction de la qualité de service à partir des KPIs d'une cellule.

**Corps de la requête :**
```json
{
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
  "sla_compliant": 1
}
```

**Réponse :**
```json
{
  "predicted_quality": "bon",
  "quality_code": 0,
  "confidence": 0.9231,
  "class_probabilities": {"bon": 0.9231, "moyen": 0.0612, "dégradé": 0.0157},
  "estimated_qos_score": 82.4,
  "is_anomaly": false,
  "anomaly_score": 0.0842,
  "model_used": "XGBoost Classifier + Isolation Forest",
  "timestamp": "2024-01-04 12:00:00"
}
```

### `GET /anomalies`
Liste des anomalies réseau détectées par l'Isolation Forest.

**Paramètres optionnels :**
- `limit` (1-100) : nombre max d'anomalies
- `slice_filter` : filtrer par tranche (eMBB, URLLC, mMTC, HC)
- `cell_filter` : filtrer par ID de cellule (ex: CELL_007)

---

## 📊 Features Engineered

Le pipeline de prétraitement génère automatiquement :

| Feature | Description |
|---------|-------------|
| `hour`, `day`, `day_of_week` | Attributs temporels |
| `is_weekend` | Indicateur week-end |
| `is_peak_hour` | Heure de pointe réseau (08h-11h, 18h-22h) |
| `*_rolling_mean` | Moyenne glissante sur 5 mesures par cellule |
| `*_rolling_std` | Écart-type glissant par cellule |

---

## 🏷️ Règles SLA par Tranche 5G

| Slice | Critère "Bon" | Critère "Dégradé" |
|-------|---------------|-------------------|
| **eMBB** | Débit ≥ 750 Mbps, Latence ≤ 10 ms, Perte ≤ 0.5% | Débit < 500 Mbps OU Latence > 14 ms OU Perte > 1.5% |
| **URLLC** | Latence ≤ 2 ms, Perte ≤ 0.04% | Latence > 3 ms OU Perte > 0.10% OU Débit < 120 Mbps |
| **mMTC** | Perte ≤ 0.5%, Latence ≤ 25 ms, Débit ≥ 25 Mbps | Perte > 2% OU Latence > 40 ms OU Débit < 15 Mbps |
| **HC** | Débit ≥ 1100 Mbps, Latence ≤ 5 ms, Perte ≤ 0.3% | Débit < 800 Mbps OU Latence > 7 ms OU Perte > 0.8% |

---

## 🧪 Tests

Lancer la suite de tests complète :

```bash
# Tests unitaires (preprocessing)
pytest tests/test_preprocessing.py -v

# Tests d'intégration (API FastAPI)
pytest tests/test_api.py -v

# Tous les tests
pytest tests/ -v
```

**Couverture des tests :**
- `test_preprocessing.py` : 7 tests unitaires (détection colonnes, nettoyage, features temporels, rolling, SLA, split, pipeline)
- `test_api.py` : 24 tests d'intégration (health, metrics, predict, anomalies — cas nominaux et dégradés)

---

## 📐 Architecture Technique

```
                   ┌─────────────────────────────────────┐
                   │         data/qos_5g.csv              │
                   │  5 760 mesures · 20 cellules · 4j    │
                   └──────────────┬──────────────────────┘
                                  │
                   ┌──────────────▼──────────────────────┐
                   │       src/preprocessing.py           │
                   │  Clean · Temporal · Rolling · SLA    │
                   └──────────────┬──────────────────────┘
                                  │
           ┌──────────────────────┼──────────────────────┐
           │                      │                      │
  ┌────────▼────────┐   ┌─────────▼────────┐  ┌─────────▼────────┐
  │  XGBoost Clf    │   │  XGBoost Reg     │  │ Isolation Forest │
  │  RF Classifier  │   │  RF Regressor    │  │  Détection ~4%   │
  │  LogReg Baseline│   │  LinReg Baseline │  │  anomalies       │
  └────────┬────────┘   └─────────┬────────┘  └─────────┬────────┘
           │                      │                      │
           └──────────────────────▼──────────────────────┘
                                  │
              ┌───────────────────┴───────────────────┐
              │                                       │
   ┌──────────▼──────────┐              ┌─────────────▼──────────┐
   │   api/main.py        │              │  app/dashboard.py       │
   │   FastAPI REST API   │              │  Streamlit Dashboard    │
   │   POST /predict      │              │  KPIs · Anomalies       │
   │   GET  /anomalies    │              │  Simulateur · ML Perf   │
   │   GET  /metrics      │              │                         │
   └──────────────────────┘              └─────────────────────────┘
```

---

## 📦 Dépendances Principales

| Package | Version min | Usage |
|---------|-------------|-------|
| `fastapi` | 0.110.0 | API REST |
| `uvicorn` | 0.28.0 | Serveur ASGI |
| `pydantic` | 2.6.0 | Validation des schémas |
| `streamlit` | 1.32.0 | Dashboard interactif |
| `xgboost` | 2.0.0 | Modèles boostés |
| `scikit-learn` | 1.4.0 | ML & prétraitement |
| `pandas` | 2.2.0 | Manipulation de données |
| `plotly` | 5.19.0 | Visualisations interactives |
| `joblib` | 1.3.0 | Sérialisation des modèles |
| `pytest` | 8.0.0 | Tests automatisés |

---

## 📄 Licence

Projet académique / de démonstration — 5G QoS Intelligence Suite.
