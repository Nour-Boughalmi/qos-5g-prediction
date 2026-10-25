"""
Générateur de données KPI 5G simulées pour le projet QoS 5G.
Génère un jeu de données réaliste sur 4 jours (3 jours Train / 1 jour Test)
pour 20 cellules (macro, micro, pico) et 4 types de tranches (eMBB, URLLC, mMTC, HC).
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

def generate_5g_kpi_dataset(
    output_path: str = "data/qos_5g.csv",
    start_date: str = "2024-01-01 00:00:00",
    days: int = 4,
    samples_per_day: int = 1440,  # 1 échantillon par minute
    random_state: int = 42
):
    np.random.seed(random_state)
    start_dt = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    total_samples = days * samples_per_day
    
    cell_ids = [f"CELL_{i:03d}" for i in range(1, 21)]
    cell_types = ["macro", "micro", "pico"]
    slice_types = ["eMBB", "URLLC", "mMTC", "HC"]
    
    # Probabilités des slices
    slice_probs = [0.45, 0.25, 0.20, 0.10]
    
    records = []
    
    for i in range(total_samples):
        current_time = start_dt + timedelta(minutes=i)
        hour = current_time.hour
        cell_id = np.random.choice(cell_ids)
        cell_type = np.random.choice(cell_types, p=[0.55, 0.30, 0.15])
        slice_type = np.random.choice(slice_types, p=slice_probs)
        
        # Effet heure de pointe (charge réseau accrue entre 8h-11h et 18h-22h)
        is_peak = (8 <= hour <= 11) or (18 <= hour <= 22)
        load_factor = 1.35 if is_peak else (0.75 if 1 <= hour <= 5 else 1.0)
        
        # Simulation selon le Slice Type
        if slice_type == "eMBB":
            # Haut débit, latence modérée
            base_tp = np.random.normal(820, 110)
            throughput_mbps = max(350.0, min(1250.0, base_tp / (1.0 + (load_factor - 1.0) * 0.25)))
            latency_ms = max(4.0, np.random.normal(8.5 * load_factor, 2.0))
            packet_loss_pct = max(0.001, np.random.exponential(0.35 * load_factor))
            active_users = max(1, int(np.random.normal(45 * load_factor, 7)))
            prb_utilization_pct = max(20.0, min(100.0, np.random.normal(70 * load_factor, 12)))
            
        elif slice_type == "URLLC":
            # Très faible latence (< 3 ms), perte quasi nulle (< 0.05%)
            throughput_mbps = max(80.0, min(450.0, np.random.normal(250, 45)))
            latency_ms = max(0.5, np.random.normal(1.8 * (1.1 if is_peak else 1.0), 0.45))
            packet_loss_pct = max(0.0001, np.random.exponential(0.035 * (1.3 if is_peak else 1.0)))
            active_users = max(1, int(np.random.normal(12 * load_factor, 3)))
            prb_utilization_pct = max(25.0, min(80.0, np.random.normal(52 * load_factor, 8)))
            
        elif slice_type == "mMTC":
            # Grand nombre d'objets, faible débit, latence plus tolérante
            throughput_mbps = max(5.0, min(85.0, np.random.normal(45, 15)))
            latency_ms = max(5.0, np.random.normal(24.0, 7.5))
            packet_loss_pct = max(0.01, np.random.exponential(0.85 * load_factor))
            active_users = max(1, int(np.random.normal(200 * load_factor, 18)))
            prb_utilization_pct = max(10.0, min(65.0, np.random.normal(33 * load_factor, 8)))
            
        else:  # HC (High Capacity / Hotspot mmWave)
            throughput_mbps = max(600.0, min(1850.0, np.random.normal(1250, 180)))
            latency_ms = max(2.0, np.random.normal(4.2, 1.0))
            packet_loss_pct = max(0.001, np.random.exponential(0.18))
            active_users = max(1, int(np.random.normal(8 * load_factor, 3)))
            prb_utilization_pct = max(40.0, min(100.0, np.random.normal(78 * load_factor, 10)))
            
        # Radio conditions (RSRP: dBm, RSRQ: dB)
        # Macro couverture plus large mais rsrp variable, pico plus localisé
        rsrp_base = -85.0 if cell_type == "pico" else (-92.0 if cell_type == "micro" else -98.0)
        rsrp_dbm = round(float(np.random.normal(rsrp_base, 10.0)), 1)
        rsrp_dbm = max(-125.0, min(-65.0, rsrp_dbm))
        
        rsrq_db = round(float(np.random.normal(-11.0, 4.0)), 2)
        rsrq_db = max(-20.0, min(-3.0, rsrq_db))
        
        # Mobilité / Handover
        handover_count = int(np.random.poisson(2.5 if cell_type == "macro" else 3.8))
        
        # Injection d'anomalies réalistes (environ 4% des cas : interférence radio, coupure backhaul, congestion extrême)
        is_anomaly = np.random.rand() < 0.04
        if is_anomaly:
            anomaly_type = np.random.choice(["interference", "congestion", "handover_failure"])
            if anomaly_type == "interference":
                rsrq_db = -19.5
                rsrp_dbm = -118.0
                packet_loss_pct = min(10.0, packet_loss_pct * 8 + 2.0)
            elif anomaly_type == "congestion":
                prb_utilization_pct = 99.8
                latency_ms = latency_ms * 3.5
                throughput_mbps = throughput_mbps * 0.3
            elif anomaly_type == "handover_failure":
                handover_count = np.random.randint(8, 15)
                packet_loss_pct = min(12.0, packet_loss_pct * 6 + 1.5)

        # Calcul du respect du SLA spécifique à la tranche
        sla_compliant = 1
        if slice_type == "eMBB" and (throughput_mbps < 600.0 or latency_ms > 14.0 or packet_loss_pct > 1.8):
            sla_compliant = 0
        elif slice_type == "URLLC" and (latency_ms > 2.5 or packet_loss_pct > 0.08):
            sla_compliant = 0
        elif slice_type == "mMTC" and (packet_loss_pct > 3.0 or latency_ms > 45.0):
            sla_compliant = 0
        elif slice_type == "HC" and (throughput_mbps < 900.0 or packet_loss_pct > 0.5):
            sla_compliant = 0

        records.append({
            "timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "cell_id": cell_id,
            "cell_type": cell_type,
            "slice_type": slice_type,
            "throughput_mbps": round(float(throughput_mbps), 2),
            "latency_ms": round(float(latency_ms), 3),
            "packet_loss_pct": round(float(packet_loss_pct), 4),
            "handover_count": int(handover_count),
            "rsrp_dbm": round(float(rsrp_dbm), 1),
            "rsrq_db": round(float(rsrq_db), 2),
            "prb_utilization_pct": round(float(prb_utilization_pct), 1),
            "active_users": int(active_users),
            "sla_compliant": int(sla_compliant)
        })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset 5G généré avec succès : {len(df)} lignes sauvegardées dans {output_path}")
    print(f"Période couverte : {df['timestamp'].min()} à {df['timestamp'].max()}")
    print("Distribution des tranches (slices) :")
    print(df['slice_type'].value_counts())
    return df

if __name__ == "__main__":
    generate_5g_kpi_dataset()
