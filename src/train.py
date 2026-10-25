"""
Script d'entraînement et de sauvegarde des modèles prédictifs QoS 5G.
Modèles entraînés :
1. Baseline : Régression Linéaire (score continu) et Régression Logistique (classification)
2. Random Forest (Classifier et Regressor)
3. XGBoost (Classifier et Regressor)
4. Isolation Forest (Détection non-supervisée d'anomalies radio/réseau)

Toutes les métriques (RMSE, R², F1, Recall, Feature Importances) sont calculées et sauvegardées.
Les artefacts sont sérialisés en .joblib dans le dossier models/.
"""

import os
import sys
import json
import logging
from typing import Dict, Any

# Garantit que la racine du projet est dans sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from xgboost import XGBClassifier, XGBRegressor

from src.preprocessing import QoSPreprocessor, temporal_train_test_split, detect_extra_columns
from src.evaluate import evaluate_regression, evaluate_classification, extract_feature_importances

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = "models"


def train_and_evaluate(data_path: str = "data/qos_5g.csv") -> Dict[str, Any]:
    """
    Pipeline complet d'entraînement, évaluation et sauvegarde.
    """
    logger.info("Chargement du jeu de données depuis : %s", data_path)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Le fichier de données '{data_path}' n'existe pas.")

    df_raw = pd.read_csv(data_path)
    logger.info("Données brutes chargées : %d lignes, %d colonnes", len(df_raw), len(df_raw.columns))
    
    # 1. Détection automatique des colonnes additionnelles
    extra_cols = detect_extra_columns(df_raw)
    
    # 2. Split chronologique (3 jours Train / 1 jour Test)
    train_df, test_df = temporal_train_test_split(df_raw, train_days=3, test_days=1)
    
    # 3. Prétraitement
    preprocessor = QoSPreprocessor(rolling_window=5)
    X_train, y_train_cls, y_train_reg = preprocessor.fit_transform(train_df)
    X_test, y_test_cls, y_test_reg = preprocessor.transform(test_df)
    
    feature_names = preprocessor.feature_names
    logger.info("Nombre total de caractéristiques (features) : %d", len(feature_names))

    os.makedirs(MODELS_DIR, exist_ok=True)
    summary_report: Dict[str, Any] = {
        "metadata": {
            "train_samples": int(len(train_df)),
            "test_samples": int(len(test_df)),
            "features_count": int(len(feature_names)),
            "feature_names": feature_names,
            "extra_columns_detected": extra_cols,
            "train_period": [str(train_df['timestamp'].min()), str(train_df['timestamp'].max())],
            "test_period": [str(test_df['timestamp'].min()), str(test_df['timestamp'].max())]
        },
        "regression_metrics": {},
        "classification_metrics": {},
        "anomaly_detection": {},
        "feature_importances": {}
    }

    # ==============================================================
    # 1. Baseline : Régression Linéaire (pour le score continu)
    # ==============================================================
    logger.info("--- Entraînement Baseline Régression Linéaire ---")
    lr = LinearRegression()
    lr.fit(X_train, y_train_reg)
    y_pred_reg_lr = lr.predict(X_test)
    metrics_lr_reg = evaluate_regression(y_test_reg, y_pred_reg_lr, model_name="Baseline Linear Regression")
    summary_report["regression_metrics"]["baseline_linear_regression"] = metrics_lr_reg
    logger.info("Baseline Linear Regression -> RMSE: %.4f | R²: %.4f", metrics_lr_reg["rmse"], metrics_lr_reg["r2"])

    # Baseline classification (Régression Logistique)
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train, y_train_cls)
    y_pred_cls_lr = log_reg.predict(X_test)
    metrics_lr_cls = evaluate_classification(y_test_cls, y_pred_cls_lr, model_name="Baseline Logistic Regression")
    summary_report["classification_metrics"]["baseline_logistic_regression"] = metrics_lr_cls

    # ==============================================================
    # 2. Random Forest (Classifier & Regressor)
    # ==============================================================
    logger.info("--- Entraînement Random Forest ---")
    rf_clf = RandomForestClassifier(n_estimators=120, max_depth=12, random_state=42, n_jobs=-1)
    rf_clf.fit(X_train, y_train_cls)
    y_pred_cls_rf = rf_clf.predict(X_test)
    metrics_rf_cls = evaluate_classification(y_test_cls, y_pred_cls_rf, model_name="Random Forest Classifier")
    summary_report["classification_metrics"]["random_forest"] = metrics_rf_cls
    logger.info("Random Forest Classifier -> F1-Macro: %.4f | Recall-Macro: %.4f | Accuracy: %.4f",
                metrics_rf_cls["f1_macro"], metrics_rf_cls["recall_macro"], metrics_rf_cls["accuracy"])

    rf_reg = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_reg.fit(X_train, y_train_reg)
    y_pred_reg_rf = rf_reg.predict(X_test)
    metrics_rf_reg = evaluate_regression(y_test_reg, y_pred_reg_rf, model_name="Random Forest Regressor")
    summary_report["regression_metrics"]["random_forest"] = metrics_rf_reg

    # ==============================================================
    # 3. XGBoost (Classifier & Regressor)
    # ==============================================================
    logger.info("--- Entraînement XGBoost ---")
    xgb_clf = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.08,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    )
    xgb_clf.fit(X_train, y_train_cls)
    y_pred_cls_xgb = xgb_clf.predict(X_test)
    metrics_xgb_cls = evaluate_classification(y_test_cls, y_pred_cls_xgb, model_name="XGBoost Classifier")
    summary_report["classification_metrics"]["xgboost"] = metrics_xgb_cls
    logger.info("XGBoost Classifier -> F1-Macro: %.4f | Recall-Macro: %.4f | Accuracy: %.4f",
                metrics_xgb_cls["f1_macro"], metrics_xgb_cls["recall_macro"], metrics_xgb_cls["accuracy"])

    xgb_reg = XGBRegressor(n_estimators=120, max_depth=6, learning_rate=0.08, random_state=42, n_jobs=-1)
    xgb_reg.fit(X_train, y_train_reg)
    y_pred_reg_xgb = xgb_reg.predict(X_test)
    metrics_xgb_reg = evaluate_regression(y_test_reg, y_pred_reg_xgb, model_name="XGBoost Regressor")
    summary_report["regression_metrics"]["xgboost"] = metrics_xgb_reg
    logger.info("XGBoost Regressor -> RMSE: %.4f | R²: %.4f", metrics_xgb_reg["rmse"], metrics_xgb_reg["r2"])

    # ==============================================================
    # 4. Isolation Forest (Détection d'anomalies 5G)
    # ==============================================================
    logger.info("--- Entraînement Isolation Forest (Anomalies 5G) ---")
    iso_forest = IsolationForest(
        n_estimators=120,
        contamination=0.04,  # Taux d'anomalies injecté ~ 4%
        random_state=42,
        n_jobs=-1
    )
    iso_forest.fit(X_train)
    
    # Prédiction : -1 = anomalie, 1 = normal
    iso_preds_train = iso_forest.predict(X_train)
    iso_preds_test = iso_forest.predict(X_test)
    anomaly_rate_train = float(np.mean(iso_preds_train == -1))
    anomaly_rate_test = float(np.mean(iso_preds_test == -1))
    
    summary_report["anomaly_detection"] = {
        "model": "Isolation Forest",
        "contamination_param": 0.04,
        "anomalies_detected_train": int(np.sum(iso_preds_train == -1)),
        "anomaly_rate_train": round(anomaly_rate_train, 4),
        "anomalies_detected_test": int(np.sum(iso_preds_test == -1)),
        "anomaly_rate_test": round(anomaly_rate_test, 4)
    }
    logger.info("Isolation Forest -> Anomalies Test détectées : %d (%.2f%%)",
                int(np.sum(iso_preds_test == -1)), anomaly_rate_test * 100)

    # ==============================================================
    # 5. Extraction des Feature Importances
    # ==============================================================
    rf_imp = extract_feature_importances(rf_clf, feature_names, top_n=15).to_dict(orient="records")
    xgb_imp = extract_feature_importances(xgb_clf, feature_names, top_n=15).to_dict(orient="records")
    summary_report["feature_importances"]["random_forest"] = rf_imp
    summary_report["feature_importances"]["xgboost"] = xgb_imp

    # ==============================================================
    # 6. Sauvegarde des Artefacts (.joblib)
    # ==============================================================
    logger.info("Sauvegarde des modèles dans le dossier '%s'...", MODELS_DIR)
    joblib.dump(preprocessor, os.path.join(MODELS_DIR, "preprocessor.joblib"))
    joblib.dump(lr, os.path.join(MODELS_DIR, "baseline_linear_reg.joblib"))
    joblib.dump(log_reg, os.path.join(MODELS_DIR, "baseline_logistic_clf.joblib"))
    joblib.dump(rf_clf, os.path.join(MODELS_DIR, "random_forest_clf.joblib"))
    joblib.dump(rf_reg, os.path.join(MODELS_DIR, "random_forest_reg.joblib"))
    joblib.dump(xgb_clf, os.path.join(MODELS_DIR, "xgboost_clf.joblib"))
    joblib.dump(xgb_reg, os.path.join(MODELS_DIR, "xgboost_reg.joblib"))
    joblib.dump(iso_forest, os.path.join(MODELS_DIR, "isolation_forest.joblib"))

    # Sauvegarde des métriques au format JSON
    metrics_path = os.path.join(MODELS_DIR, "evaluation_summary.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=4, ensure_ascii=False)
        
    logger.info("Entraînement terminé avec succès ! Métriques enregistrées dans %s", metrics_path)
    return summary_report


if __name__ == "__main__":
    train_and_evaluate()
