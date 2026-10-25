"""
Dashboard Streamlit pour la supervision et prédiction de Qualité de Service (QoS) 5G.
Fonctionnalités :
- Courbes KPI interactives temporelles par cellule
- Filtres dynamiques par Tranche de réseau (Slice) et par Cellule
- Alertes et détection d'anomalies en temps réel (Isolation Forest)
- Formulaire de prédiction interactive (XGBoost & Random Forest)
- Synthèse des métriques d'évaluation et matrice de confusion
"""

import os
import sys
import json
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import joblib

# Assure que la racine est accessible
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.preprocessing import QoSPreprocessor, INV_QUALITY_LABEL_MAP

# Configuration de la page
st.set_page_config(
    page_title="5G QoS Intelligence Suite",
    page_icon="📶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé élégant et moderne
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.8));
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        margin-bottom: 12px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #38BDF8;
    }
    .metric-title {
        font-size: 14px;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-bon {
        background-color: #10B981;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
    }
    .badge-moyen {
        background-color: #F59E0B;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
    }
    .badge-degrade {
        background-color: #EF4444;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0px 0px;
        padding: 10px 20px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

DATA_PATH = os.path.join(BASE_DIR, "data", "qos_5g.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")


@st.cache_data(ttl=600)
def load_dataset():
    """Charge et prépare les données 5G."""
    if not os.path.exists(DATA_PATH):
        st.error(f"Fichier de données introuvable : {DATA_PATH}")
        return pd.DataFrame()
    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


@st.cache_resource
def load_ml_models():
    """Charge les modèles entraînés et le préprocesseur."""
    try:
        prep = joblib.load(os.path.join(MODELS_DIR, "preprocessor.joblib"))
        xgb_clf = joblib.load(os.path.join(MODELS_DIR, "xgboost_clf.joblib"))
        rf_clf = joblib.load(os.path.join(MODELS_DIR, "random_forest_clf.joblib"))
        xgb_reg = joblib.load(os.path.join(MODELS_DIR, "xgboost_reg.joblib"))
        iso = joblib.load(os.path.join(MODELS_DIR, "isolation_forest.joblib"))
        
        summary_path = os.path.join(MODELS_DIR, "evaluation_summary.json")
        summary = {}
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
                
        return {
            "preprocessor": prep,
            "xgb_clf": xgb_clf,
            "rf_clf": rf_clf,
            "xgb_reg": xgb_reg,
            "iso": iso,
            "summary": summary
        }
    except Exception as e:
        st.warning(f"Impossible de charger tous les modèles : {e}")
        return None


df_raw = load_dataset()
ml_bundle = load_ml_models()

# ==============================================================
# Barre latérale : Filtres & Métadonnées
# ==============================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/5g.png", width=64)
    st.title("5G QoS Dashboard")
    st.caption("Supervision de la Qualité de Service & Prédiction IA")
    
    st.markdown("---")
    st.subheader("🔍 Filtres Réseau")
    
    if not df_raw.empty:
        available_slices = ["Toutes"] + sorted(df_raw["slice_type"].unique().tolist())
        selected_slice = st.selectbox("Type de tranche (Slice)", available_slices, index=0)
        
        available_cells = ["Toutes"] + sorted(df_raw["cell_id"].unique().tolist())
        selected_cell = st.selectbox("Cellule cible", available_cells, index=0)
        
        available_types = ["Tous"] + sorted(df_raw["cell_type"].unique().tolist())
        selected_type = st.selectbox("Type d'antenne", available_types, index=0)

        # Filtrage dynamique des données
        filtered_df = df_raw.copy()
        if selected_slice != "Toutes":
            filtered_df = filtered_df[filtered_df["slice_type"] == selected_slice]
        if selected_cell != "Toutes":
            filtered_df = filtered_df[filtered_df["cell_id"] == selected_cell]
        if selected_type != "Tous":
            filtered_df = filtered_df[filtered_df["cell_type"] == selected_type]
            
        st.caption(f"📊 {len(filtered_df):,} mesures sélectionnées sur {len(df_raw):,}")
    else:
        filtered_df = pd.DataFrame()

    st.markdown("---")
    st.info("ℹ️ **Note sur les données :** Les données KPI sont des métriques 5G simulées réalistes (3GPP Rel. 16/17).")


# ==============================================================
# En-tête principal & KPIs synthétiques
# ==============================================================
st.title("📶 Plateforme Prédictive de Qualité de Service Réseau 5G")
st.markdown("Supervision en continu des KPIs critiques, respect des SLAs par tranche et anticipation des dégradations de service.")

if not filtered_df.empty:
    avg_tp = filtered_df["throughput_mbps"].mean()
    avg_lat = filtered_df["latency_ms"].mean()
    avg_loss = filtered_df["packet_loss_pct"].mean()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Débit Moyen (Throughput)</div>
            <div class="metric-value">{avg_tp:,.1f} <span style="font-size:16px;">Mbps</span></div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Latence Moyenne</div>
            <div class="metric-value">{avg_lat:,.2f} <span style="font-size:16px;">ms</span></div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Taux de Perte Paquets</div>
            <div class="metric-value">{avg_loss:,.3f} <span style="font-size:16px;">%</span></div>
        </div>
        """, unsafe_allow_html=True)



# ==============================================================
# Onglets du Dashboard
# ==============================================================
tab1, tab2, tab3 = st.tabs([
    "📈 Analyse Temporelle des KPIs",
    "⚠️ Détection d'Anomalies & Alertes",
    "🔮 Simulateur & Prédiction QoS",
])

# --------------------------------------------------------------
# ONGLET 1 : Analyse Temporelle des KPIs
# --------------------------------------------------------------
with tab1:
    st.subheader("Évolution Chronologique des KPIs par Cellule")
    
    if filtered_df.empty:
        st.warning("Aucune donnée disponible pour les filtres sélectionnés.")
    else:
        # Échantillonnage pour la fluidité si nécessaire
        plot_df = filtered_df.tail(800) if len(filtered_df) > 800 else filtered_df

        # Graphique 1 : Débit et Latence
        fig_tp_lat = go.Figure()
        fig_tp_lat.add_trace(go.Scatter(
            x=plot_df["timestamp"],
            y=plot_df["throughput_mbps"],
            name="Débit (Mbps)",
            line=dict(color="#38BDF8", width=2),
            yaxis="y1"
        ))
        fig_tp_lat.add_trace(go.Scatter(
            x=plot_df["timestamp"],
            y=plot_df["latency_ms"],
            name="Latence (ms)",
            line=dict(color="#F43F5E", width=1.8, dash="dot"),
            yaxis="y2"
        ))
        fig_tp_lat.update_layout(
            title="Débit (Mbps) vs Latence (ms) au cours du temps",
            template="plotly_dark",
            height=420,
            xaxis=dict(title="Horodatage"),
            yaxis=dict(title="Débit (Mbps)", title_font=dict(color="#38BDF8")),
            yaxis2=dict(title="Latence (ms)", title_font=dict(color="#F43F5E"), overlaying="y", side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_tp_lat, use_container_width=True)

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            # Perte de paquets
            fig_loss = px.line(
                plot_df,
                x="timestamp",
                y="packet_loss_pct",
                color="cell_id" if selected_cell == "Toutes" else None,
                title="Taux de Perte de Paquets (%)",
                template="plotly_dark",
                height=350
            )
            fig_loss.add_hline(y=1.5, line_dash="dash", line_color="orange", annotation_text="Seuil Alerte (1.5%)")
            st.plotly_chart(fig_loss, use_container_width=True)

        with col_g2:
            # Couverture Radio (RSRP / RSRQ)
            # Clip active_users >= 1 car Plotly exige size > 0
            scatter_df = plot_df.copy()
            if "active_users" in scatter_df.columns:
                scatter_df["active_users"] = scatter_df["active_users"].clip(lower=1)

            fig_radio = px.scatter(
                scatter_df,
                x="rsrp_dbm",
                y="rsrq_db",
                color="slice_type",
                size="active_users" if "active_users" in scatter_df.columns else None,
                hover_data=["cell_id", "throughput_mbps", "latency_ms"],
                title="Qualité Radio : RSRQ (dB) vs RSRP (dBm)",
                template="plotly_dark",
                height=350
            )
            st.plotly_chart(fig_radio, use_container_width=True)



# --------------------------------------------------------------
# ONGLET 2 : Détection d'Anomalies & Alertes
# --------------------------------------------------------------
with tab2:
    st.subheader("Système d'Alerte et Détection d'Anomalies (Isolation Forest)")
    st.markdown("L'algorithme **Isolation Forest** détecte de manière non supervisée les dérives atypiques de charge, pannes de cellules et interférences sévères.")

    if ml_bundle and "iso" in ml_bundle and "preprocessor" in ml_bundle:
        prep: QoSPreprocessor = ml_bundle["preprocessor"]
        iso = ml_bundle["iso"]
        
        # Détection sur le sous-ensemble actuel
        sample_eval = filtered_df.tail(600).copy().reset_index(drop=True)
        if not sample_eval.empty:
            X_scaled, _, _ = prep.transform(sample_eval)
            preds = iso.predict(X_scaled)
            scores = iso.decision_function(X_scaled)
            sample_eval["is_anomaly"] = (preds == -1)
            sample_eval["anomaly_score"] = np.round(scores, 4)

            anomalies_found = sample_eval[sample_eval["is_anomaly"]].copy()
            
            c_a1, c_a2, c_a3 = st.columns(3)
            with c_a1:
                st.metric("Mesures Analysées", f"{len(sample_eval):,}")
            with c_a2:
                st.metric("Anomalies Détectées", f"{len(anomalies_found):,}", delta=f"{len(anomalies_found)/len(sample_eval)*100:.1f}%")
            with c_a3:
                st.metric("Cellules Impactées", f"{anomalies_found['cell_id'].nunique() if not anomalies_found.empty else 0}")

            if not anomalies_found.empty:
                st.markdown("### 🚨 Tableau des Cellules en Défaillance / Dégradées")
                # Explication du motif
                def diagnose(row):
                    reasons = []
                    if row.get("packet_loss_pct", 0) > 2.0: reasons.append("Perte paquet critique")
                    if row.get("latency_ms", 0) > 30.0: reasons.append("Latence anormale")
                    if row.get("prb_utilization_pct", 0) > 95.0: reasons.append("Saturation PRB")
                    if row.get("rsrp_dbm", 0) < -115.0: reasons.append("Signal radio faible")
                    return ", ".join(reasons) if reasons else "Déviation multivariée"

                anomalies_found["Diagnostic"] = anomalies_found.apply(diagnose, axis=1)
                
                cols_to_show = ["timestamp", "cell_id", "cell_type", "slice_type", "throughput_mbps", "latency_ms", "packet_loss_pct", "rsrp_dbm", "Diagnostic"]
                st.dataframe(
                    anomalies_found[cols_to_show].sort_values("timestamp", ascending=False),
                    use_container_width=True,
                    height=300
                )

                # Graphique des anomalies dans le temps
                fig_anom = px.scatter(
                    sample_eval,
                    x="timestamp",
                    y="throughput_mbps",
                    color="is_anomaly",
                    color_discrete_map={True: "#EF4444", False: "#38BDF8"},
                    title="Dispersion du Débit : Cellules Normales (Bleu) vs Anomalies (Rouge)",
                    template="plotly_dark",
                    height=380
                )
                st.plotly_chart(fig_anom, use_container_width=True)
            else:
                st.success("✅ Aucune anomalie détectée sur la période et le périmètre sélectionnés.")
    else:
        st.warning("Les modèles ML ne sont pas encore prêts. Entraînez les modèles avec `python src/train.py`.")


# --------------------------------------------------------------
# ONGLET 3 : Simulateur & Prédiction QoS
# --------------------------------------------------------------
with tab3:
    st.subheader("Simulateur Interactif de Prédiction de Qualité de Service")
    st.markdown("Modifiez les paramètres opérationnels d'une cellule pour estimer instantanément sa classe de qualité (`bon`, `moyen`, `dégradé`).")

    with st.form("qos_simulation_form"):
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            in_cell_id = st.selectbox("Cellule ID", [f"CELL_{i:03d}" for i in range(1, 21)], index=6)
            in_cell_type = st.selectbox("Type d'antenne", ["macro", "micro", "pico"], index=0)
            in_slice_type = st.selectbox("Type de Tranche (Slice)", ["eMBB", "URLLC", "mMTC", "HC"], index=0)

        with col_s2:
            in_tp = st.slider("Débit Mesuré (Mbps)", 1.0, 1800.0, 820.0, step=10.0)
            in_lat = st.slider("Latence (ms)", 0.5, 60.0, 7.5, step=0.5)
            in_loss = st.slider("Perte de paquets (%)", 0.0, 8.0, 0.15, step=0.05)

        with col_s3:
            in_rsrp = st.slider("RSRP (dBm)", -125.0, -65.0, -92.0, step=1.0)
            in_rsrq = st.slider("RSRQ (dB)", -20.0, -3.0, -10.0, step=0.5)
            in_prb = st.slider("Utilisation PRB (%)", 0.0, 100.0, 65.0, step=1.0)
            in_users = st.number_input("Utilisateurs Actifs", min_value=1, max_value=500, value=45)

        submit_btn = st.form_submit_button("⚡ Prédire la Qualité de Service", use_container_width=True)

    if submit_btn and ml_bundle:
        prep: QoSPreprocessor = ml_bundle["preprocessor"]
        xgb_clf = ml_bundle["xgb_clf"]
        xgb_reg = ml_bundle["xgb_reg"]
        iso = ml_bundle["iso"]

        input_data = {
            "cell_id": in_cell_id,
            "cell_type": in_cell_type,
            "slice_type": in_slice_type,
            "throughput_mbps": in_tp,
            "latency_ms": in_lat,
            "packet_loss_pct": in_loss,
            "handover_count": 2,
            "rsrp_dbm": in_rsrp,
            "rsrq_db": in_rsrq,
            "prb_utilization_pct": in_prb,
            "active_users": in_users,
            "sla_compliant": 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        X_input = prep.prepare_single_instance(input_data)
        
        # Prédictions
        pred_class = int(xgb_clf.predict(X_input)[0])
        pred_label = INV_QUALITY_LABEL_MAP.get(pred_class, "inconnu")
        probabilities = xgb_clf.predict_proba(X_input)[0]
        confidence = float(np.max(probabilities))
        est_score = float(xgb_reg.predict(X_input)[0]) if xgb_reg else 75.0
        est_score = max(0.0, min(100.0, est_score))
        
        is_anom = bool(iso.predict(X_input)[0] == -1) if iso else False

        # Affichage du résultat
        st.markdown("---")
        st.subheader("Résultats de la Prédiction")
        
        r_c1, r_c2, r_c3 = st.columns(3)
        with r_c1:
            badge_class = "badge-bon" if pred_label == "bon" else ("badge-moyen" if pred_label == "moyen" else "badge-degrade")
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Qualité de Service Estimée</div>
                <div style="margin-top:10px;">
                    <span class="{badge_class}" style="font-size:22px; padding: 6px 16px;">{pred_label.upper()}</span>
                </div>
                <div style="margin-top:8px; color:#94A3B8;">Indice de Confiance : <b>{confidence*100:.1f}%</b></div>
            </div>
            """, unsafe_allow_html=True)

        with r_c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Score QoS Global Continu</div>
                <div class="metric-value">{est_score:.1f} <span style="font-size:16px;">/ 100</span></div>
                <div style="margin-top:8px; color:#94A3B8;">Prédit par Régresseur XGBoost</div>
            </div>
            """, unsafe_allow_html=True)

        with r_c3:
            anom_color = "#EF4444" if is_anom else "#10B981"
            anom_text = "⚠️ ANOMALIE DÉTECTÉE" if is_anom else "✅ FONCTIONNEMENT NORMAL"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Statut Réseau (Isolation Forest)</div>
                <div style="font-size:18px; font-weight:700; color:{anom_color}; margin-top:8px;">{anom_text}</div>
                <div style="margin-top:8px; color:#94A3B8;">Déviation comportementale</div>
            </div>
            """, unsafe_allow_html=True)

        # Probabilités détaillées
        st.markdown("#### Répartition des Probabilités par Classe")
        prob_df = pd.DataFrame({
            "Classe": [k.capitalize() for k in INV_QUALITY_LABEL_MAP.values()],
            "Probabilité (%)": [probabilities[i] * 100 for i in range(len(INV_QUALITY_LABEL_MAP))]
        })
        fig_prob = px.bar(
            prob_df,
            x="Classe",
            y="Probabilité (%)",
            color="Classe",
            color_discrete_map={"Bon": "#10B981", "Moyen": "#F59E0B", "Dégradé": "#EF4444"},
            template="plotly_dark",
            height=260
        )
        st.plotly_chart(fig_prob, use_container_width=True)



