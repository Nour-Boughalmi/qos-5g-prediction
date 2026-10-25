"""
Module d'évaluation des modèles de qualité de service 5G.
Calcule :
- Métriques de régression : RMSE, MAE, R²
- Métriques de classification : Accuracy, F1-macro, F1-weighted, Recall (par classe et macro)
- Importance des variables (Feature Importances)
- Matrice de confusion
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
    classification_report
)

def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> Dict[str, float]:
    """
    Calcule RMSE, MAE et R² pour un modèle de régression.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    metrics = {
        "model": model_name,
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4)
    }
    return metrics


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Calcule Accuracy, F1-score, Recall, Précision et Matrice de confusion.
    """
    if class_names is None:
        class_names = ["bon", "moyen", "dégradé"]

    acc = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    recall_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    precision_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    
    # Recall par classe individuelle
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    recall_by_class = {
        f"recall_{class_names[i] if i < len(class_names) else f'class_{i}'}": round(float(val), 4)
        for i, val in enumerate(per_class_recall)
    }

    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)

    metrics = {
        "model": model_name,
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "recall_macro": round(recall_macro, 4),
        "precision_macro": round(precision_macro, 4),
        "confusion_matrix": cm,
        "classification_report": report,
        **recall_by_class
    }
    return metrics


def extract_feature_importances(model: Any, feature_names: List[str], top_n: int = 15) -> pd.DataFrame:
    """
    Extrait l'importance des variables pour les modèles arborescents (Random Forest, XGBoost)
    ou les coefficients pour la régression linéaire.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        # Valeur absolue des coefficients
        coef = model.coef_
        if coef.ndim > 1:
            importances = np.mean(np.abs(coef), axis=0)
        else:
            importances = np.abs(coef)
    else:
        return pd.DataFrame()

    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    return df_imp.head(top_n)
