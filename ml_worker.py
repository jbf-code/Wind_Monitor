"""
Wind Monitor - ML Worker
========================
Computes per-turbine anomaly scores, trend detection, and cross-turbine
peer benchmarking for all turbines in the fleet.

Three algorithms (all using numpy only — no scipy dependency):
  A. Anomaly detection   — z-score vs each turbine's own 24 h rolling baseline
  B. Trend detection     — linear regression slope (numpy.polyfit) over last 24 h
  C. Peer comparison     — percentile rank vs same make+model fleet peers

Results are stored in the `ml_insights` table (one upserted row per turbine)
and a plain-text summary block is generated for injection into Claude prompts.

Usage:
  Standalone:  python ml_worker.py
  Via Flask:   POST /api/ml/run  (triggers this in a background thread)
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta

import numpy as np
from sqlalchemy import create_engine, text as sa_text
from sqlalchemy.orm import sessionmaker

# Allow standalone execution from the project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format='[ML] %(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────

ML_METRICS = [
    'power_output_kw',
    'wind_speed_ms',
    'bearing_temp_c',
    'gearbox_temp_c',
    'generator_temp_c',
    'oil_temp_c',
    'vibration_gearbox_x',
    'vibration_gearbox_y',
    'vibration_gearbox_z',
    'vibration_bearing',
    'acoustic_level_db',
    'ambient_temp_c',
]

TREND_METRICS = [
    'bearing_temp_c',
    'gearbox_temp_c',
    'vibration_gearbox_x',
    'vibration_bearing',
    'acoustic_level_db',
]

METRIC_UNITS = {
    'bearing_temp_c':      '°C/hour',
    'gearbox_temp_c':      '°C/hour',
    'vibration_gearbox_x': 'g/hour',
    'vibration_bearing':   'g/hour',
    'acoustic_level_db':   'dB/hour',
}

# Degradation metrics: HIGH percentile = WORST in fleet
DEGRADATION_METRICS = {
    'bearing_temp_c', 'gearbox_temp_c', 'generator_temp_c', 'oil_temp_c',
    'vibration_gearbox_x', 'vibration_gearbox_y', 'vibration_gearbox_z',
    'vibration_bearing', 'acoustic_level_db',
}

ANOMALY_Z_THRESHOLD  = 2.5   # |z| above this = anomaly flag
TREND_RAPID_RISE     = 0.5   # slope/hour above this = "rapid_rise"
TREND_SLOPE_MIN      = 0.05  # |slope|/hour below this = "stable"
PEER_WORST_PCT       = 90    # percentile >= this is flagged for degradation metrics
LOOKBACK_HOURS       = 24    # window for baseline, trend, and peer snapshot
DATA_DAYS            = 7     # how far back to load sensor readings
MIN_READINGS         = 30    # minimum data points before computing any statistic


# ─── Database Session (standalone mode) ───────────────────────────────────────

def _get_standalone_session():
    """Create a fresh SQLAlchemy session from environment variables."""
    db_url = os.environ.get(
        'DATABASE_URL',
        'sqlite:///' + os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            os.environ.get('DATABASE_PATH', 'wind_monitor.db'),
        )
    )
    if db_url.startswith('postgres://'):
        db_url = db_url.replace('postgres://', 'postgresql+psycopg://', 1)
    elif db_url.startswith('postgresql://') and '+psycopg' not in db_url:
        db_url = db_url.replace('postgresql://', 'postgresql+psycopg://', 1)

    engine = create_engine(db_url, pool_pre_ping=True)
    Session = sessionmaker(bind=engine)
    return Session(), engine


# ─── Data Loading ─────────────────────────────────────────────────────────────

def _load_sensor_data(session, since: datetime) -> dict:
    """
    Load all sensor readings for all turbines since `since`.
    Returns: {turbine_id: {"timestamps": [datetime, ...], "values": {metric: np.ndarray}}}
    """
    metrics_sql = ', '.join(ML_METRICS)
    sql = sa_text(f"""
        SELECT turbine_id, timestamp, {metrics_sql}
        FROM sensor_readings
        WHERE timestamp >= :since
        ORDER BY turbine_id, timestamp
    """)
    rows = session.execute(sql, {"since": since}).fetchall()
    log.info(f"Loaded {len(rows)} sensor rows from last {DATA_DAYS} days")

    raw: dict = {}
    for row in rows:
        tid = row[0]
        ts  = row[1]
        if tid not in raw:
            raw[tid] = {"timestamps": [], "values": {m: [] for m in ML_METRICS}}
        raw[tid]["timestamps"].append(ts)
        for i, metric in enumerate(ML_METRICS):
            val = row[2 + i]
            raw[tid]["values"][metric].append(val if val is not None else np.nan)

    # Convert lists → numpy arrays
    data: dict = {}
    for tid, d in raw.items():
        data[tid] = {
            "timestamps": d["timestamps"],   # keep as Python list of datetime for slicing
            "values": {m: np.array(d["values"][m], dtype=float) for m in ML_METRICS},
        }

    return data


# ─── Algorithm A: Anomaly Detection (z-score) ─────────────────────────────────

def _compute_anomaly_scores(td: dict) -> dict:
    """
    Per-metric z-score of the most recent reading vs the turbine's own 24 h baseline.

    Baseline window : t_latest − 25 h  →  t_latest − 1 h
    Recent window   : last 10 minutes (median used to suppress transient spikes)

    Returns: {metric: float z-score or None}
    """
    timestamps = td["timestamps"]
    if not timestamps:
        return {m: None for m in ML_METRICS}

    t_latest         = max(timestamps)
    t_baseline_start = t_latest - timedelta(hours=LOOKBACK_HOURS + 1)
    t_baseline_end   = t_latest - timedelta(hours=1)
    t_recent_start   = t_latest - timedelta(minutes=10)

    scores = {}
    for metric in ML_METRICS:
        vals = td["values"][metric]

        baseline_vals = np.array([
            vals[i] for i, ts in enumerate(timestamps)
            if t_baseline_start <= ts <= t_baseline_end
        ])
        recent_vals = np.array([
            vals[i] for i, ts in enumerate(timestamps)
            if ts >= t_recent_start
        ])

        baseline_vals = baseline_vals[~np.isnan(baseline_vals)]
        recent_vals   = recent_vals[~np.isnan(recent_vals)]

        if len(baseline_vals) < MIN_READINGS or len(recent_vals) == 0:
            scores[metric] = None
            continue

        mean = float(np.mean(baseline_vals))
        std  = float(np.std(baseline_vals))

        if std < 1e-6:   # constant signal (e.g. turbine offline)
            scores[metric] = None
            continue

        current = float(np.median(recent_vals))
        scores[metric] = round((current - mean) / std, 3)

    return scores


# ─── Algorithm B: Trend Detection (linear regression via numpy.polyfit) ────────

def _compute_trends(td: dict) -> dict:
    """
    Linear regression slope (numpy.polyfit degree-1) over the last LOOKBACK_HOURS
    for each TREND_METRIC.

    Slope is expressed in units-per-hour.
    Direction classification:
      |slope| < TREND_SLOPE_MIN  →  "stable"
      slope >= TREND_RAPID_RISE  →  "rapid_rise"
      slope >  0                 →  "rising"
      slope <  0                 →  "falling"

    Returns: {metric: {"slope_per_hour": float, "direction": str} or None}
    """
    timestamps = td["timestamps"]
    if not timestamps:
        return {m: None for m in TREND_METRICS}

    t_latest     = max(timestamps)
    window_start = t_latest - timedelta(hours=LOOKBACK_HOURS)

    trends = {}
    for metric in TREND_METRICS:
        vals = td["values"][metric]

        # Slice to the 24 h window
        idx_mask  = [i for i, ts in enumerate(timestamps) if ts >= window_start]
        if not idx_mask:
            trends[metric] = None
            continue

        ts_window  = [timestamps[i] for i in idx_mask]
        val_window = vals[np.array(idx_mask)]

        # Remove NaN pairs
        valid_mask   = ~np.isnan(val_window)
        ts_clean     = [ts_window[i] for i in range(len(ts_window)) if valid_mask[i]]
        vals_clean   = val_window[valid_mask]

        if len(vals_clean) < MIN_READINGS:
            trends[metric] = None
            continue

        # Convert timestamps to fractional hours since window start
        t0      = ts_clean[0]
        t_hours = np.array([(ts - t0).total_seconds() / 3600.0 for ts in ts_clean])

        try:
            coeffs = np.polyfit(t_hours, vals_clean, 1)
        except np.linalg.LinAlgError:
            trends[metric] = None
            continue

        slope = float(coeffs[0])
        abs_s = abs(slope)

        if abs_s < TREND_SLOPE_MIN:
            direction = 'stable'
        elif slope >= TREND_RAPID_RISE:
            direction = 'rapid_rise'
        elif slope > 0:
            direction = 'rising'
        else:
            direction = 'falling'

        trends[metric] = {
            "slope_per_hour": round(slope, 4),
            "direction": direction,
        }

    return trends


# ─── Algorithm C: Peer Comparison (percentile ranking) ────────────────────────

def _compute_peer_ranks(all_data: dict, turbine_id: int, turbine_groups: dict) -> dict:
    """
    For each ML_METRIC, compute this turbine's percentile rank vs same make+model peers.

    Percentile = count(peers with median_recent <= my_median_recent) / n_peers × 100.
    High percentile = worst in fleet for DEGRADATION_METRICS.
    Requires at least 2 turbines in the peer group.

    Returns: {metric: int percentile (0–100) or None}
    """
    my_group = turbine_groups.get(turbine_id)
    if not my_group:
        return {}

    peer_ids = [tid for tid, grp in turbine_groups.items() if grp == my_group]
    if len(peer_ids) < 2:
        return {}

    ranks = {}
    for metric in ML_METRICS:
        peer_medians: dict = {}
        for pid in peer_ids:
            if pid not in all_data:
                continue
            ptd = all_data[pid]
            if not ptd["timestamps"]:
                continue
            t_latest = max(ptd["timestamps"])
            t_recent = t_latest - timedelta(minutes=10)
            idx = [i for i, ts in enumerate(ptd["timestamps"]) if ts >= t_recent]
            if not idx:
                continue
            recent_vals = ptd["values"][metric][np.array(idx)]
            recent_vals = recent_vals[~np.isnan(recent_vals)]
            if len(recent_vals) > 0:
                peer_medians[pid] = float(np.median(recent_vals))

        if len(peer_medians) < 2 or turbine_id not in peer_medians:
            ranks[metric] = None
            continue

        my_val    = peer_medians[turbine_id]
        all_vals  = sorted(peer_medians.values())
        count_lte = sum(1 for v in all_vals if v <= my_val)
        ranks[metric] = int(round(100 * count_lte / len(all_vals)))

    return ranks


# ─── Health Score ──────────────────────────────────────────────────────────────

def _compute_health_score(anomaly_scores: dict, trends: dict, peer_ranks: dict) -> float:
    """
    Composite 0–100 score (100 = perfect health). Deductions:
      Anomaly:    −8 × min(|z| / threshold, 2.0)  per flagged metric
      Rapid rise: −10  per trend metric
      Rising:     −4   per trend metric
      Worst peer: −5   per degradation metric at ≥ PEER_WORST_PCT percentile
    """
    score = 100.0

    for metric, z in (anomaly_scores or {}).items():
        if z is not None and abs(z) > ANOMALY_Z_THRESHOLD:
            severity = min(abs(z) / ANOMALY_Z_THRESHOLD, 2.0)
            score -= 8.0 * severity

    for metric, t in (trends or {}).items():
        if t is None:
            continue
        if t["direction"] == "rapid_rise":
            score -= 10.0
        elif t["direction"] == "rising":
            score -= 4.0

    for metric, pct in (peer_ranks or {}).items():
        if metric in DEGRADATION_METRICS and pct is not None and pct >= PEER_WORST_PCT:
            score -= 5.0

    return round(max(0.0, min(100.0, score)), 1)


# ─── Claude Prompt Summary ────────────────────────────────────────────────────

def _build_ml_summary(
    turbine_name: str,
    computed_at: datetime,
    health_score,
    anomaly_scores: dict,
    trends: dict,
    peer_ranks: dict,
    peer_group_label: str,
    peer_group_size: int,
) -> str:
    """
    Generate the plain-text block prepended to Claude's diagnostic prompt.
    Format matches the specification in the implementation plan.
    """
    lines = [
        f"=== MACHINE LEARNING ANALYSIS (computed {computed_at.strftime('%Y-%m-%d %H:%M UTC')}) ===",
        f"Health Score: {health_score}/100" if health_score is not None else "Health Score: N/A",
        "",
    ]

    # Anomaly flags
    flagged = {
        m: z for m, z in (anomaly_scores or {}).items()
        if z is not None and abs(z) > ANOMALY_Z_THRESHOLD
    }
    lines.append(f"Anomaly Flags ({len(flagged)} metrics flagged):")
    if flagged:
        for m, z in sorted(flagged.items(), key=lambda x: abs(x[1]), reverse=True):
            sign      = '+' if z > 0 else ''
            direction = 'significantly above' if z > 0 else 'significantly below'
            lines.append(f"  - {m}: z-score {sign}{z:.1f} ({direction} turbine's own 24h baseline)")
    else:
        lines.append("  None")
    lines.append("")

    # Trend analysis
    non_stable = {
        m: t for m, t in (trends or {}).items()
        if t is not None and t["direction"] != "stable"
    }
    lines.append(f"Trend Analysis (last {LOOKBACK_HOURS}h):")
    if non_stable:
        for m, t in non_stable.items():
            unit = METRIC_UNITS.get(m, '/hour')
            sign = '+' if t["slope_per_hour"] > 0 else ''
            lines.append(
                f"  - {m}: {t['direction'].upper()} ({sign}{t['slope_per_hour']:.3f} {unit} slope)"
            )
    else:
        lines.append("  All trend metrics stable")
    lines.append("")

    # Peer comparison
    if peer_ranks and peer_group_size > 1:
        n_peers = peer_group_size - 1
        lines.append(f"Peer Comparison (vs {n_peers} other {peer_group_label} turbine{'s' if n_peers > 1 else ''}):")
        for m, pct in peer_ranks.items():
            if pct is not None:
                lines.append(f"  - {m}: {pct}th percentile")
    lines.append("")

    # Key ML finding
    lines.append(f"Key ML Finding: {_derive_key_finding(flagged, trends or {}, peer_ranks or {})}")
    lines.append("===")

    return "\n".join(lines)


def _derive_key_finding(flagged: dict, trends: dict, peer_ranks: dict) -> str:
    """Single-sentence summary of the most significant ML finding."""
    has_bearing = (
        'bearing_temp_c' in flagged
        or (trends.get('bearing_temp_c') or {}).get('direction') in ('rising', 'rapid_rise')
    )
    has_gearbox_vib = any(
        m in flagged for m in ('vibration_gearbox_x', 'vibration_gearbox_y', 'vibration_gearbox_z')
    )
    has_gearbox_temp = 'gearbox_temp_c' in flagged
    has_acoustic     = 'acoustic_level_db' in flagged

    if has_bearing and has_gearbox_vib:
        return "Bearing degradation signature with corroborating vibration and temperature trends."
    if has_bearing:
        return "Elevated bearing temperature trend detected; monitor for early-stage bearing failure."
    if has_gearbox_vib and has_gearbox_temp:
        return "Gearbox degradation signature: concurrent vibration and temperature anomalies."
    if has_gearbox_vib:
        return "Gearbox vibration anomaly detected; possible gear wear or misalignment."
    if has_acoustic and has_gearbox_temp:
        return "Elevated acoustic and gearbox temperature signals; possible lubrication issue."
    if not flagged:
        return "No significant anomalies detected. All metrics within normal operating range."
    worst = max(flagged.items(), key=lambda x: abs(x[1]))[0]
    return f"Anomaly detected in {worst.replace('_', ' ')}; further investigation recommended."


# ─── Main Orchestration ───────────────────────────────────────────────────────

def run_ml_analysis(session=None) -> dict:
    """
    Main entry point.

    Parameters
    ----------
    session : SQLAlchemy session, optional
        If None, a standalone session is created from environment variables.
        If provided (Flask app context), used directly — caller is responsible
        for commit/rollback.

    Returns
    -------
    dict with keys: status, turbines_processed, computed_at  (or status, message on error)
    """
    standalone = session is None
    standalone_engine = None

    if standalone:
        log.info("Running ML worker in STANDALONE mode")
        session, standalone_engine = _get_standalone_session()
    else:
        log.info("Running ML worker in FLASK CONTEXT mode")

    try:
        now   = datetime.utcnow()
        since = now - timedelta(days=DATA_DAYS)

        # Load turbine metadata and build peer groups
        turbines = session.execute(
            sa_text("SELECT id, name, make, model FROM turbines")
        ).fetchall()

        turbine_groups: dict = {}       # {turbine_id: "Make Model"}
        turbine_names:  dict = {}       # {turbine_id: name}
        group_sizes:    dict = {}       # {"Make Model": count}

        for row in turbines:
            tid, name, make, model = row[0], row[1], row[2], row[3]
            grp = f"{make} {model}"
            turbine_groups[tid] = grp
            turbine_names[tid]  = name
            group_sizes[grp]    = group_sizes.get(grp, 0) + 1

        log.info(f"Fleet: {len(turbines)} turbines across {len(group_sizes)} make/model groups")

        # Load all sensor data once for the full fleet
        all_data = _load_sensor_data(session, since)

        results = {}
        for turbine_id, grp_label in turbine_groups.items():
            log.info(f"  [{turbine_names[turbine_id]}] computing ML insight...")

            td = all_data.get(turbine_id, {
                "timestamps": [],
                "values": {m: np.array([], dtype=float) for m in ML_METRICS},
            })

            anomaly_scores = _compute_anomaly_scores(td)
            trends         = _compute_trends(td)
            peer_ranks     = _compute_peer_ranks(all_data, turbine_id, turbine_groups)
            health_score   = _compute_health_score(anomaly_scores, trends, peer_ranks)

            ml_summary = _build_ml_summary(
                turbine_name     = turbine_names[turbine_id],
                computed_at      = now,
                health_score     = health_score,
                anomaly_scores   = anomaly_scores,
                trends           = trends,
                peer_ranks       = peer_ranks,
                peer_group_label = grp_label,
                peer_group_size  = group_sizes.get(grp_label, 1),
            )

            raw_blob = json.dumps({
                "computed_at":    now.isoformat(),
                "health_score":   health_score,
                "anomaly_scores": anomaly_scores,
                "trends":         trends,
                "peer_ranks":     peer_ranks,
            })

            # Upsert: check for existing row, then UPDATE or INSERT
            existing = session.execute(
                sa_text("SELECT id FROM ml_insights WHERE turbine_id = :tid"),
                {"tid": turbine_id},
            ).fetchone()

            params = {
                "turbine_id":     turbine_id,
                "computed_at":    now,
                "health_score":   health_score,
                "anomaly_scores": json.dumps(anomaly_scores),
                "trend_data":     json.dumps(trends),
                "peer_ranks":     json.dumps(peer_ranks),
                "ml_summary":     ml_summary,
                "raw_results":    raw_blob,
            }

            if existing:
                session.execute(sa_text("""
                    UPDATE ml_insights SET
                        computed_at    = :computed_at,
                        health_score   = :health_score,
                        anomaly_scores = :anomaly_scores,
                        trend_data     = :trend_data,
                        peer_ranks     = :peer_ranks,
                        ml_summary     = :ml_summary,
                        raw_results    = :raw_results
                    WHERE turbine_id   = :turbine_id
                """), params)
            else:
                session.execute(sa_text("""
                    INSERT INTO ml_insights
                        (turbine_id, computed_at, health_score, anomaly_scores,
                         trend_data, peer_ranks, ml_summary, raw_results)
                    VALUES
                        (:turbine_id, :computed_at, :health_score, :anomaly_scores,
                         :trend_data, :peer_ranks, :ml_summary, :raw_results)
                """), params)

            results[turbine_id] = {"health_score": health_score}
            log.info(
                f"    health={health_score}  "
                f"anomalies={sum(1 for z in anomaly_scores.values() if z and abs(z) > ANOMALY_Z_THRESHOLD)}  "
                f"non-stable-trends={sum(1 for t in trends.values() if t and t['direction'] != 'stable')}"
            )

        session.commit()
        log.info(f"ML analysis complete — {len(results)} turbines processed.")
        return {
            "status":             "ok",
            "turbines_processed": len(results),
            "computed_at":        now.isoformat(),
        }

    except Exception as exc:
        log.error(f"ML analysis failed: {exc}", exc_info=True)
        try:
            session.rollback()
        except Exception:
            pass
        return {"status": "error", "message": str(exc)}

    finally:
        if standalone:
            session.close()
            if standalone_engine:
                standalone_engine.dispose()


# ─── Standalone Entry Point ───────────────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    result = run_ml_analysis()
    print(json.dumps(result, indent=2))
