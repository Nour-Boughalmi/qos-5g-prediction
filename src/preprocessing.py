"""
Module de prétraitement des données pour le projet QoS 5G.
Gère :
- Nettoyage et typage des KPIs
- Détection automatique des colonnes supplémentaires
- Extraction des attributs temporels (heure, jour, week-end, heure de pointe)
- Calcul des moyennes glissantes par cellule (rolling statistics)
- Attribution de la cible de qualité (bon / moyen / dégradé) selon les SLAs de chaque slice
- Séparation chronologique (split 3 jours train / 1 jour test)
- Encodage et normalisation
"""

import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Colonnes de base attendues
BASE_COLUMNS = [
    "timestamp",
    "cell_id",
    "cell_type",
    "slice_type",
    "throughput_mbps",
    "latency_ms",
    "packet_loss_pct",
    "handover_count",
    "rsrp_dbm",
    "rsrq_db",
]

# Mapping numérique pour la classification
QUALITY_LABEL_MAP = {
    "bon": 0,
    "moyen": 1,
    "dégradé": 2
}
INV_QUALITY_LABEL_MAP = {v: k for k, v in QUALITY_LABEL_MAP.items()}


def detect_extra_columns(df: pd.DataFrame) -> List[str]:
    """
    Détecte automatiquement les colonnes présentes dans le DataFrame
    qui ne font pas partie des colonnes de base standard.
    """
    extra_cols = [c for c in df.columns if c not in BASE_COLUMNS and c not in ["qos_quality", "qos_target", "qos_score"]]
    logger.info("Colonnes supplémentaires détectées automatiquement (%d) : %s", len(extra_cols), extra_cols)
    return extra_cols


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie les données brutes :
    - Conversion de la date
    - Suppression des doublons
    - Traitement des valeurs aberrantes ou manquantes
    - Tri chronologique
    """
    data = df.copy()
    
    # Conversion timestamp
    if "timestamp" in data.columns:
        data["timestamp"] = pd.to_datetime(data["timestamp"])
        data = data.sort_values("timestamp").reset_index(drop=True)
        
    # Suppression des doublons stricts
    initial_len = len(data)
    data = data.drop_duplicates()
    if len(data) < initial_len:
        logger.info("Suppression de %d doublons", initial_len - len(data))

    # Imputation simple pour valeurs manquantes
    num_cols = data.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if data[col].isnull().any():
            median_val = data[col].median()
            data[col] = data[col].fillna(median_val)
            logger.info("Imputation médiane pour %s : %s", col, median_val)
            
    cat_cols = data.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        if data[col].isnull().any():
            mode_val = data[col].mode()[0]
            data[col] = data[col].fillna(mode_val)
            logger.info("Imputation mode pour %s : %s", col, mode_val)
            
    # Bornes physiques logiques
    if "throughput_mbps" in data.columns:
        data["throughput_mbps"] = data["throughput_mbps"].clip(lower=0.1)
    if "latency_ms" in data.columns:
        data["latency_ms"] = data["latency_ms"].clip(lower=0.1)
    if "packet_loss_pct" in data.columns:
        data["packet_loss_pct"] = data["packet_loss_pct"].clip(lower=0.0, upper=100.0)

    return data


def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrait l'heure, le jour, le jour de la semaine et les indicateurs
    de week-end et d'heure de pointe (busy hour).
    """
    data = df.copy()
    if "timestamp" not in data.columns:
        return data

    ts = pd.to_datetime(data["timestamp"])
    data["hour"] = ts.dt.hour
    data["day"] = ts.dt.day
    data["day_of_week"] = ts.dt.dayofweek
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)
    
    # Heures de pointe réseau typiques : 08h-11h et 18h-22h
    data["is_peak_hour"] = data["hour"].apply(
        lambda h: 1 if (8 <= h <= 11) or (18 <= h <= 22) else 0
    )
    return data


def compute_rolling_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Calcule des moyennes glissantes (rolling means) par cellule (cell_id)
    sur les KPIs critiques (throughput, latency, packet_loss, etc.).
    """
    data = df.copy()
    if "cell_id" not in data.columns:
        return data
        
    kpis_to_roll = ["throughput_mbps", "latency_ms", "packet_loss_pct"]
    if "prb_utilization_pct" in data.columns:
        kpis_to_roll.append("prb_utilization_pct")
    if "active_users" in data.columns:
        kpis_to_roll.append("active_users")

    for kpi in kpis_to_roll:
        if kpi in data.columns:
            # Moyenne glissante passée par cellule
            col_name = f"{kpi}_rolling_mean"
            data[col_name] = (
                data.groupby("cell_id")[kpi]
                .transform(lambda s: s.rolling(window=window, min_periods=1).mean())
            )
            # Écart-type glissant pour capter la variabilité
            col_std = f"{kpi}_rolling_std"
            data[col_std] = (
                data.groupby("cell_id")[kpi]
                .transform(lambda s: s.rolling(window=window, min_periods=1).std().fillna(0.0))
            )
            
    return data


def assign_slice_qos_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attribue la cible de qualité de service ('bon', 'moyen', 'dégradé')
    en fonction de règles et seuils stricts propres à chaque type de tranche (slice) 5G.
    
    Règles par tranche (SLA métier 5G) :
    - eMBB (Mobile Broadband) :
        * bon : Débit >= 750 Mbps ET Latence <= 10 ms ET Perte <= 0.5%
        * dégradé : Débit < 500 Mbps OU Latence > 14 ms OU Perte > 1.5%
        * moyen : entre les deux
    - URLLC (Ultra-Reliable Low Latency) :
        * bon : Latence <= 2.0 ms ET Perte <= 0.04%
        * dégradé : Latence > 3.0 ms OU Perte > 0.10% OU Débit < 120 Mbps
        * moyen : entre les deux
    - mMTC (Massive IoT) :
        * bon : Perte <= 0.5% ET Latence <= 25 ms ET Débit >= 25 Mbps
        * dégradé : Perte > 2.0% OU Latence > 40 ms OU Débit < 15 Mbps
        * moyen : entre les deux
    - HC (High Capacity) :
        * bon : Débit >= 1100 Mbps ET Latence <= 5.0 ms ET Perte <= 0.3%
        * dégradé : Débit < 800 Mbps OU Latence > 7.0 ms OU Perte > 0.8%
        * moyen : entre les deux
    """
    data = df.copy()

    def evaluate_row(row):
        slice_type = str(row.get("slice_type", "eMBB")).strip()
        tp = float(row.get("throughput_mbps", 0.0))
        lat = float(row.get("latency_ms", 0.0))
        loss = float(row.get("packet_loss_pct", 0.0))

        if slice_type == "eMBB":
            if tp >= 750.0 and lat <= 10.0 and loss <= 0.5:
                return "bon"
            elif tp < 500.0 or lat > 14.0 or loss > 1.5:
                return "dégradé"
            else:
                return "moyen"

        elif slice_type == "URLLC":
            if lat <= 2.0 and loss <= 0.04:
                return "bon"
            elif lat > 3.0 or loss > 0.10 or tp < 120.0:
                return "dégradé"
            else:
                return "moyen"

        elif slice_type == "mMTC":
            if loss <= 0.5 and lat <= 25.0 and tp >= 25.0:
                return "bon"
            elif loss > 2.0 or lat > 40.0 or tp < 15.0:
                return "dégradé"
            else:
                return "moyen"

        elif slice_type == "HC":
            if tp >= 1100.0 and lat <= 5.0 and loss <= 0.3:
                return "bon"
            elif tp < 800.0 or lat > 7.0 or loss > 0.8:
                return "dégradé"
            else:
                return "moyen"

        else:
            # Règle composite générique par défaut
            if tp >= 500.0 and lat <= 15.0 and loss <= 0.5:
                return "bon"
            elif tp < 200.0 or lat > 30.0 or loss > 2.0:
                return "dégradé"
            else:
                return "moyen"

    data["qos_quality"] = data.apply(evaluate_row, axis=1)
    data["qos_target"] = data["qos_quality"].map(QUALITY_LABEL_MAP)
    
    # Score composite continu de QoS (0 à 100) pour la régression
    data["qos_score"] = compute_continuous_qos_score(data)
    
    return data


def compute_continuous_qos_score(df: pd.DataFrame) -> pd.Series:
    """
    Calcule un indice composite continu de QoS (0 à 100)
    en intégrant débit relatif, latence normalisée, perte et radio.
    """
    # Normalisation min-max douce
    tp_score = np.clip(df["throughput_mbps"] / 1200.0 * 100.0, 0, 100)
    lat_score = np.clip(100.0 - (df["latency_ms"] / 40.0 * 100.0), 0, 100)
    loss_score = np.clip(100.0 - (df["packet_loss_pct"] / 3.0 * 100.0), 0, 100)
    
    # Poids équilibrés
    score = (tp_score * 0.40) + (lat_score * 0.35) + (loss_score * 0.25)
    return score.round(2)


def temporal_train_test_split(
    df: pd.DataFrame,
    train_days: int = 3,
    test_days: int = 1
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split chronologique sans fuite de données :
    - Train : les 3 premiers jours
    - Test : le dernier jour
    """
    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values("timestamp").reset_index(drop=True)
    min_date = data["timestamp"].min()
    split_cutoff = min_date + pd.Timedelta(days=train_days)
    
    train_df = data[data["timestamp"] < split_cutoff].copy().reset_index(drop=True)
    test_df = data[data["timestamp"] >= split_cutoff].copy().reset_index(drop=True)
    
    logger.info(
        "Split chronologique appliqué (Cutoff = %s) : Train=%d lignes (%.1f%%), Test=%d lignes (%.1f%%)",
        split_cutoff,
        len(train_df),
        100.0 * len(train_df) / len(data),
        len(test_df),
        100.0 * len(test_df) / len(data)
    )
    return train_df, test_df


class QoSPreprocessor:
    """
    Pipeline complet de prétraitement pour la prédiction et l'inférence.
    Gère l'encodage, la normalisation et la détection dynamique de colonnes.
    """
    def __init__(self, rolling_window: int = 5):
        self.rolling_window = rolling_window
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.categorical_columns: List[str] = ["cell_id", "cell_type", "slice_type"]
        self.extra_columns: List[str] = []
        self.fitted: bool = False

    def fit(self, df: pd.DataFrame):
        """Ajuste les encodeurs et le scaler sur le jeu d'entraînement."""
        cleaned = clean_data(df)
        self.extra_columns = detect_extra_columns(cleaned)
        with_time = extract_temporal_features(cleaned)
        with_rolling = compute_rolling_features(with_time, window=self.rolling_window)
        with_target = assign_slice_qos_target(with_rolling)

        # Encodage des catégories
        transformed = with_target.copy()
        for col in self.categorical_columns:
            if col in transformed.columns:
                le = LabelEncoder()
                # Cast en string pour robustesse
                transformed[col] = le.fit_transform(transformed[col].astype(str))
                self.label_encoders[col] = le

        # Sélection des features
        drop_cols = ["timestamp", "qos_quality", "qos_target", "qos_score"]
        feature_cols = [c for c in transformed.columns if c not in drop_cols]
        self.feature_names = feature_cols
        
        # Fit scaler
        X = transformed[self.feature_names].values
        self.scaler.fit(X)
        self.fitted = True
        logger.info("QoSPreprocessor fitted avec succès sur %d features : %s", len(self.feature_names), self.feature_names)
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Transforme un DataFrame en matrices X (normalisée), y_class (si présent), y_reg (si présent).
        """
        if not self.fitted:
            raise RuntimeError("Le preprocessor doit être 'fit' avant 'transform'.")
            
        cleaned = clean_data(df)
        with_time = extract_temporal_features(cleaned)
        with_rolling = compute_rolling_features(with_time, window=self.rolling_window)
        
        # Calcul de la cible si possible
        if "qos_quality" not in with_rolling.columns and "slice_type" in with_rolling.columns:
            with_target = assign_slice_qos_target(with_rolling)
        else:
            with_target = with_rolling

        transformed = with_target.copy()
        
        # Encodage catégoriel avec gestion des valeurs inconnues
        for col, le in self.label_encoders.items():
            if col in transformed.columns:
                known_classes = set(le.classes_)
                transformed[col] = transformed[col].astype(str).apply(
                    lambda x: x if x in known_classes else le.classes_[0]
                )
                transformed[col] = le.transform(transformed[col].astype(str))

        # Vérification des colonnes manquantes
        for col in self.feature_names:
            if col not in transformed.columns:
                transformed[col] = 0.0

        X = transformed[self.feature_names].values
        X_scaled = self.scaler.transform(X)

        y_class = transformed["qos_target"].values if "qos_target" in transformed.columns else None
        y_reg = transformed["qos_score"].values if "qos_score" in transformed.columns else None

        return X_scaled, y_class, y_reg

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Fit et transformation directe."""
        return self.fit(df).transform(df)

    def prepare_single_instance(self, instance_dict: Dict[str, Any]) -> np.ndarray:
        """
        Prépare une requête unitaire (dictionnaire) pour l'API de prédiction.
        """
        if not self.fitted:
            raise RuntimeError("Le preprocessor n'est pas encore entraîné.")

        # Construction d'un DataFrame à 1 ligne
        df_single = pd.DataFrame([instance_dict])
        
        # Si pas de timestamp, assigner maintenant
        if "timestamp" not in df_single.columns:
            df_single["timestamp"] = pd.Timestamp.now()
            
        X_scaled, _, _ = self.transform(df_single)
        return X_scaled
