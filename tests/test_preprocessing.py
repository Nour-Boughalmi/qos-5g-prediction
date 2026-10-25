"""
Tests unitaires pour le module src/preprocessing.py.
Vérifie :
- Détection automatique des colonnes
- Nettoyage et complétion des données
- Extraction des attributs temporels
- Calcul des moyennes glissantes
- Règles d'attribution de la cible QoS par slice
- Découpage chronologique sans fuite
- Pipeline QoSPreprocessor
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.preprocessing import (
    detect_extra_columns,
    clean_data,
    extract_temporal_features,
    compute_rolling_features,
    assign_slice_qos_target,
    temporal_train_test_split,
    QoSPreprocessor
)


@pytest.fixture
def sample_kpi_dataframe():
    """Crée un petit jeu de données de test sur 4 jours."""
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    records = []
    
    # 4 jours de données
    for d in range(4):
        for h in range(6):  # quelques échantillons par jour
            t = base_time + timedelta(days=d, hours=h * 4)
            records.append({
                "timestamp": t.strftime("%Y-%m-%d %H:%M:%S"),
                "cell_id": "CELL_001" if h % 2 == 0 else "CELL_002",
                "cell_type": "macro",
                "slice_type": "eMBB" if d % 2 == 0 else "URLLC",
                "throughput_mbps": 850.0 if d % 2 == 0 else 250.0,
                "latency_ms": 7.0 if d % 2 == 0 else 1.8,
                "packet_loss_pct": 0.10 if d % 2 == 0 else 0.02,
                "handover_count": 2,
                "rsrp_dbm": -90.0,
                "rsrq_db": -10.0,
                # 3 colonnes supplémentaires
                "prb_utilization_pct": 55.0,
                "active_users": 40,
                "sla_compliant": 1
            })
            
    return pd.DataFrame(records)


def test_detect_extra_columns(sample_kpi_dataframe):
    """Vérifie que les 3 colonnes additionnelles sont correctement détectées."""
    extra_cols = detect_extra_columns(sample_kpi_dataframe)
    assert "prb_utilization_pct" in extra_cols
    assert "active_users" in extra_cols
    assert "sla_compliant" in extra_cols
    assert "throughput_mbps" not in extra_cols


def test_clean_data():
    """Vérifie le nettoyage, conversion datetime et traitement des valeurs manquantes."""
    raw = pd.DataFrame({
        "timestamp": ["2024-01-01 10:00:00", "2024-01-01 10:00:00", "2024-01-01 09:00:00"],
        "throughput_mbps": [500.0, 500.0, np.nan],
        "latency_ms": [-5.0, -5.0, 10.0],
        "packet_loss_pct": [0.2, 0.2, 120.0]
    })
    cleaned = clean_data(raw)
    
    # Doit avoir supprimé le doublon exact
    assert len(cleaned) == 2
    # Tri chronologique vérifié
    assert cleaned["timestamp"].iloc[0] < cleaned["timestamp"].iloc[1]
    # Clamping physique des valeurs
    assert (cleaned["latency_ms"] >= 0.1).all()
    assert (cleaned["packet_loss_pct"] <= 100.0).all()
    # NaN imputé
    assert not cleaned["throughput_mbps"].isnull().any()


def test_extract_temporal_features(sample_kpi_dataframe):
    """Vérifie l'extraction des caractéristiques temporelles."""
    df_feat = extract_temporal_features(sample_kpi_dataframe)
    assert "hour" in df_feat.columns
    assert "day_of_week" in df_feat.columns
    assert "is_weekend" in df_feat.columns
    assert "is_peak_hour" in df_feat.columns
    assert df_feat["hour"].between(0, 23).all()
    assert set(df_feat["is_peak_hour"].unique()).issubset({0, 1})


def test_compute_rolling_features(sample_kpi_dataframe):
    """Vérifie le calcul des rolling features par cellule."""
    df_roll = compute_rolling_features(sample_kpi_dataframe, window=3)
    assert "throughput_mbps_rolling_mean" in df_roll.columns
    assert "latency_ms_rolling_mean" in df_roll.columns
    assert not df_roll["throughput_mbps_rolling_mean"].isnull().any()


def test_assign_slice_qos_target():
    """Vérifie que les règles de seuil métier par slice s'appliquent correctement."""
    df_slices = pd.DataFrame([
        # eMBB bon
        {"slice_type": "eMBB", "throughput_mbps": 850.0, "latency_ms": 7.0, "packet_loss_pct": 0.1},
        # eMBB dégradé
        {"slice_type": "eMBB", "throughput_mbps": 400.0, "latency_ms": 20.0, "packet_loss_pct": 2.5},
        # URLLC bon
        {"slice_type": "URLLC", "throughput_mbps": 200.0, "latency_ms": 1.5, "packet_loss_pct": 0.02},
        # URLLC dégradé
        {"slice_type": "URLLC", "throughput_mbps": 100.0, "latency_ms": 4.5, "packet_loss_pct": 0.20},
        # mMTC bon
        {"slice_type": "mMTC", "throughput_mbps": 35.0, "latency_ms": 15.0, "packet_loss_pct": 0.2},
        # mMTC dégradé
        {"slice_type": "mMTC", "throughput_mbps": 10.0, "latency_ms": 50.0, "packet_loss_pct": 3.5},
    ])
    
    result = assign_slice_qos_target(df_slices)
    qualities = result["qos_quality"].tolist()
    
    assert qualities[0] == "bon"
    assert qualities[1] == "dégradé"
    assert qualities[2] == "bon"
    assert qualities[3] == "dégradé"
    assert qualities[4] == "bon"
    assert qualities[5] == "dégradé"
    assert "qos_score" in result.columns


def test_temporal_train_test_split(sample_kpi_dataframe):
    """Vérifie le split 3 jours train / 1 jour test sans fuite chronologique."""
    train_df, test_df = temporal_train_test_split(sample_kpi_dataframe, train_days=3, test_days=1)
    
    assert len(train_df) > 0
    assert len(test_df) > 0
    assert train_df["timestamp"].max() < test_df["timestamp"].min()


def test_qos_preprocessor_pipeline(sample_kpi_dataframe):
    """Vérifie que le préprocesseur s'ajuste et transforme sans erreur."""
    train_df, test_df = temporal_train_test_split(sample_kpi_dataframe, train_days=3, test_days=1)
    
    prep = QoSPreprocessor(rolling_window=2)
    X_train, y_train_cls, y_train_reg = prep.fit_transform(train_df)
    X_test, y_test_cls, y_test_reg = prep.transform(test_df)
    
    assert X_train.shape[0] == len(train_df)
    assert X_test.shape[0] == len(test_df)
    assert X_train.shape[1] == len(prep.feature_names)
    assert len(y_train_cls) == len(train_df)
    assert len(y_train_reg) == len(train_df)

    # Test requête unitaire
    single_req = {
        "cell_id": "CELL_001",
        "cell_type": "macro",
        "slice_type": "eMBB",
        "throughput_mbps": 800.0,
        "latency_ms": 8.0,
        "packet_loss_pct": 0.2,
        "handover_count": 1,
        "rsrp_dbm": -95.0,
        "rsrq_db": -12.0
    }
    X_single = prep.prepare_single_instance(single_req)
    assert X_single.shape == (1, len(prep.feature_names))
