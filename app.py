"""
Wind Monitor - Flask Application
Core web service for wind turbine farm monitoring dashboard.
"""
import os
import threading
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request, redirect, url_for, session, abort
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from sqlalchemy import func
from database import db, Turbine, SensorReading, TurbineEvent

app = Flask(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-change-in-production')
_db_url = os.environ.get('DATABASE_URL', 'sqlite:///' + os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    os.environ.get('DATABASE_PATH', 'wind_monitor.db')
))
# Railway injects postgres:// but SQLAlchemy requires postgresql://
if _db_url.startswith('postgres://'):
    _db_url = _db_url.replace('postgres://', 'postgresql://', 1)
app.config['SQLALCHEMY_DATABASE_URI'] = _db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_pre_ping': True,
    'pool_size': 5,
    'max_overflow': 10,
}

# AI API keys (stubs — replace with real keys when ready)
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')
GEMINI_API_KEY    = os.environ.get('GOOGLE_GEMINI_API_KEY', '')
ADMIN_PASSWORD    = os.environ.get('ADMIN_PASSWORD', 'JBF_2026')

db.init_app(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# ─── Simple User Model ─────────────────────────────────────────────────────────
class AdminUser(UserMixin):
    def get_id(self): return "admin"

@login_manager.user_loader
def load_user(user_id):
    if user_id == "admin":
        return AdminUser()
    return None

# ─── DB Init & Seeding (background thread) ────────────────────────────────────
_seeded = False

def init_and_seed():
    global _seeded
    with app.app_context():
        db.create_all()
        from database import Turbine
        if Turbine.query.count() == 0:
            print("Starting database seed (background)...")
            from seed_data import seed_database
            seed_database(app)
        _seeded = True

# Run seeding in background so the app starts immediately
_seed_thread = threading.Thread(target=init_and_seed, daemon=True)
_seed_thread.start()

# ─── Auth Routes ──────────────────────────────────────────────────────────────
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        if username == 'admin' and password == ADMIN_PASSWORD:
            user = AdminUser()
            login_user(user, remember=True)
            return redirect(url_for('dashboard'))
        error = 'invalid'
    return render_template('login.html', error=error)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# ─── Page Routes ──────────────────────────────────────────────────────────────
@app.route('/')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/turbine/<int:turbine_id>')
@login_required
def turbine_detail(turbine_id):
    turbine = Turbine.query.get_or_404(turbine_id)
    return render_template('turbine_detail.html', turbine=turbine)

@app.route('/status')
def status():
    """Health check endpoint for Railway."""
    return jsonify({"status": "ok", "seeded": _seeded, "ts": datetime.utcnow().isoformat()})

# ─── API: Fleet Overview ───────────────────────────────────────────────────────
@app.route('/api/fleet')
@login_required
def api_fleet():
    """All turbines with latest reading."""
    turbines = Turbine.query.all()
    result = []
    for t in turbines:
        latest = (SensorReading.query
                  .filter_by(turbine_id=t.id)
                  .order_by(SensorReading.timestamp.desc())
                  .first())
        events_open = TurbineEvent.query.filter_by(turbine_id=t.id, resolved=False).count()
        result.append({
            "id": t.id,
            "name": t.name,
            "make": t.make,
            "model": t.model,
            "serial_number": t.serial_number,
            "rated_power_kw": t.rated_power_kw,
            "lat": t.latitude,
            "lng": t.longitude,
            "status": t.status,
            "open_events": events_open,
            "latest": _reading_dict(latest) if latest else None,
        })
    return jsonify(result)

# ─── API: Single Turbine ───────────────────────────────────────────────────────
@app.route('/api/turbine/<int:turbine_id>')
@login_required
def api_turbine(turbine_id):
    t = Turbine.query.get_or_404(turbine_id)
    events = (TurbineEvent.query
              .filter_by(turbine_id=turbine_id)
              .order_by(TurbineEvent.timestamp.desc())
              .limit(20).all())
    latest = (SensorReading.query
              .filter_by(turbine_id=turbine_id)
              .order_by(SensorReading.timestamp.desc())
              .first())
    return jsonify({
        "id": t.id,
        "name": t.name,
        "make": t.make,
        "model": t.model,
        "serial_number": t.serial_number,
        "rated_power_kw": t.rated_power_kw,
        "hub_height_m": t.hub_height_m,
        "rotor_diameter_m": t.rotor_diameter_m,
        "lat": t.latitude,
        "lng": t.longitude,
        "status": t.status,
        "build_date": t.build_date.isoformat(),
        "last_service_date": t.last_service_date.isoformat(),
        "notes": t.notes,
        "latest": _reading_dict(latest) if latest else None,
        "events": [_event_dict(e) for e in events],
    })

# ─── API: Time-series Chart Data ──────────────────────────────────────────────
@app.route('/api/turbine/<int:turbine_id>/chart')
@login_required
def api_chart(turbine_id):
    """
    Returns time-series data for a given metric.
    Query params:
      metric  - field name in SensorReading
      range   - '24h' | '7d' | '21d' (default '24h')
      res     - 'auto' | 'min' | 'hour' (default 'auto')
    """
    metric = request.args.get('metric', 'power_output_kw')
    rng    = request.args.get('range', '24h')
    res    = request.args.get('res', 'auto')

    now = datetime.utcnow()
    if rng == '24h':
        since = now - timedelta(hours=24)
        resolution = 'min'
    elif rng == '7d':
        since = now - timedelta(days=7)
        resolution = 'hour'
    else:  # 21d
        since = now - timedelta(days=21)
        resolution = 'hour'

    if res != 'auto':
        resolution = res

    allowed_metrics = {
        'power_output_kw', 'wind_speed_ms', 'rotor_rpm', 'pitch_angle_deg',
        'nacelle_temp_c', 'bearing_temp_c', 'gearbox_temp_c', 'generator_temp_c',
        'oil_temp_c', 'ambient_temp_c',
        'vibration_gearbox_x', 'vibration_gearbox_y', 'vibration_gearbox_z',
        'vibration_bearing', 'acoustic_level_db',
        'nacelle_humidity_pct', 'grid_voltage_v', 'grid_frequency_hz',
    }
    if metric not in allowed_metrics:
        abort(400, f"Invalid metric: {metric}")

    col = getattr(SensorReading, metric)
    rows = (SensorReading.query
            .with_entities(SensorReading.timestamp, col)
            .filter(
                SensorReading.turbine_id == turbine_id,
                SensorReading.timestamp >= since,
                SensorReading.resolution == resolution,
            )
            .order_by(SensorReading.timestamp.asc())
            .all())

    return jsonify({
        "metric": metric,
        "range": rng,
        "resolution": resolution,
        "data": [
            {"t": r[0].isoformat(), "v": round(r[1], 3) if r[1] is not None else None}
            for r in rows
        ]
    })

# ─── API: Fleet Power Summary ──────────────────────────────────────────────────
@app.route('/api/fleet/power')
@login_required
def api_fleet_power():
    """Aggregated total fleet power. ?range=48h (default) or 7d.
    Both ranges use resolution='min' rows (last 7 days are all minute-resolution).
    48h groups by hour; 7d groups by day.
    """
    range_param = request.args.get('range', '48h')
    now = datetime.utcnow()
    is_sqlite = 'sqlite' in app.config['SQLALCHEMY_DATABASE_URI']

    if range_param == '7d':
        since = now - timedelta(days=7)
        if is_sqlite:
            bucket_expr = func.strftime('%Y-%m-%dT00:00:00', SensorReading.timestamp).label('bucket')
        else:
            bucket_expr = func.date_trunc('day', SensorReading.timestamp).label('bucket')
    else:
        since = now - timedelta(hours=48)
        if is_sqlite:
            bucket_expr = func.strftime('%Y-%m-%dT%H:00:00', SensorReading.timestamp).label('bucket')
        else:
            bucket_expr = func.date_trunc('hour', SensorReading.timestamp).label('bucket')

    rows = (db.session.query(
                bucket_expr,
                func.sum(SensorReading.power_output_kw).label('total_kw'),
                func.avg(SensorReading.wind_speed_ms).label('avg_wind'),
            )
            .filter(
                SensorReading.timestamp >= since,
                SensorReading.resolution == 'min',
            )
            .group_by(bucket_expr)
            .order_by(bucket_expr)
            .all())
    return jsonify([
        {"t": r[0].isoformat() if hasattr(r[0], 'isoformat') else str(r[0]),
         "total_kw": round(r[1] or 0, 1),
         "avg_wind": round(r[2] or 0, 2)}
        for r in rows
    ])

# ─── API: Debug ───────────────────────────────────────────────────────────────
@app.route('/api/debug/data')
@login_required
def api_debug_data():
    """Debug: show resolution counts and timestamp ranges in the DB."""
    from sqlalchemy import text
    rows = db.session.execute(text(
        "SELECT resolution, COUNT(*) as cnt, MIN(timestamp) as oldest, MAX(timestamp) as newest "
        "FROM sensor_readings GROUP BY resolution ORDER BY resolution"
    )).fetchall()
    now = datetime.utcnow()
    return jsonify({
        "server_utc": now.isoformat(),
        "resolutions": [
            {"resolution": r[0], "count": r[1],
             "oldest": str(r[2]), "newest": str(r[3])}
            for r in rows
        ]
    })

# ─── API: Events ──────────────────────────────────────────────────────────────
@app.route('/api/events')
@login_required
def api_events():
    """Recent events across all turbines."""
    limit = int(request.args.get('limit', 50))
    severity = request.args.get('severity')
    q = TurbineEvent.query.order_by(TurbineEvent.timestamp.desc())
    if severity:
        q = q.filter_by(severity=severity)
    events = q.limit(limit).all()
    return jsonify([_event_dict(e) for e in events])

# ─── API: AI Analysis ──────────────────────────────────────────────────────────
@app.route('/api/ai/analyze/<int:turbine_id>', methods=['POST'])
@login_required
def api_ai_analyze(turbine_id):
    """Call Claude API with latest sensor data and events for diagnostic analysis."""
    if not ANTHROPIC_API_KEY:
        return jsonify({
            "status": "stub",
            "message": "AI integration not configured. Set ANTHROPIC_API_KEY environment variable.",
            "turbine_id": turbine_id,
        }), 200

    turbine = Turbine.query.get_or_404(turbine_id)

    # Fetch last 100 minute-resolution readings
    readings = (SensorReading.query
                .filter_by(turbine_id=turbine_id, resolution='min')
                .order_by(SensorReading.timestamp.desc())
                .limit(100).all())
    readings = list(reversed(readings))

    # Fetch last 20 unresolved events
    events = (TurbineEvent.query
              .filter_by(turbine_id=turbine_id, resolved=False)
              .order_by(TurbineEvent.timestamp.desc())
              .limit(20).all())

    def fmt_readings(rs):
        lines = []
        for r in rs:
            lines.append(
                f"{r.timestamp.strftime('%Y-%m-%d %H:%M')} | "
                f"power={r.power_output_kw:.1f}kW wind={r.wind_speed_ms:.1f}m/s "
                f"rpm={r.rotor_rpm:.1f} pitch={r.pitch_angle_deg:.1f}deg "
                f"bearing_t={r.bearing_temp_c:.1f}C gearbox_t={r.gearbox_temp_c:.1f}C "
                f"gen_t={r.generator_temp_c:.1f}C oil_t={r.oil_temp_c:.1f}C "
                f"vib_x={r.vibration_gearbox_x:.3f}g vib_y={r.vibration_gearbox_y:.3f}g "
                f"vib_z={r.vibration_gearbox_z:.3f}g vib_bearing={r.vibration_bearing:.3f}g "
                f"acoustic={r.acoustic_level_db:.1f}dB"
            )
        return "\n".join(lines)

    def fmt_events(evs):
        if not evs:
            return "No active alerts."
        lines = []
        for e in evs:
            lines.append(
                f"{e.timestamp.strftime('%Y-%m-%d %H:%M')} [{e.severity.upper()}] "
                f"{e.category}/{e.code}: {e.message_en}"
            )
        return "\n".join(lines)

    readings_text = fmt_readings(readings)
    events_text = fmt_events(events)

    prompt = f"""You are an expert wind turbine condition monitoring engineer. Analyze the following sensor data and respond with ONLY valid JSON — no prose, no markdown fences.

Turbine: {turbine.name} | {turbine.make} {turbine.model} | rated {turbine.rated_power_kw} kW | status: {turbine.status}
Build: {turbine.build_date} | Last service: {turbine.last_service_date}

Active alerts:
{events_text}

Recent sensor readings (last {len(readings)} minutes):
{readings_text}

Respond with this exact JSON structure:
{{
  "verdict": "OK | MONITOR | ACTION REQUIRED | CRITICAL",
  "summary": "One sentence overall condition assessment.",
  "next_step": "Single most important action to take right now.",
  "pf_position": "early warning | developing | imminent | none",
  "key_findings": ["finding 1", "finding 2", "finding 3"],
  "root_cause": "Brief root cause assessment.",
  "recommended_actions": {{
    "immediate": "...",
    "within_7_days": "...",
    "scheduled": "..."
  }},
  "risk_if_ignored": "Brief consequence statement."
}}"""

    try:
        import json as json_lib
        from anthropic import Anthropic
        client = Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        raw_text = response.content[0].text.strip()

        # Strip markdown code fences if Claude wrapped the JSON
        clean_text = raw_text
        if clean_text.startswith("```"):
            clean_text = clean_text.split("\n", 1)[-1]  # drop first line (```json)
        if clean_text.endswith("```"):
            clean_text = clean_text.rsplit("```", 1)[0]
        clean_text = clean_text.strip()

        # Parse structured JSON from Claude
        try:
            structured = json_lib.loads(clean_text)
        except json_lib.JSONDecodeError:
            # Fallback: return raw text if JSON parsing fails
            structured = None

        # Build full log entry stored in DB
        log_payload = json_lib.dumps({
            "prompt_summary": {
                "turbine": turbine.name,
                "readings_count": len(readings),
                "active_alerts": len(events),
                "readings_from": readings[0].timestamp.isoformat() if readings else None,
                "readings_to": readings[-1].timestamp.isoformat() if readings else None,
            },
            "response_raw": raw_text,
            "response_parsed": structured,
        }, indent=2)

        # Log AI analysis as a TurbineEvent
        ai_event = TurbineEvent(
            turbine_id=turbine_id,
            timestamp=datetime.utcnow(),
            severity="info",
            category="ai_analysis",
            code="AI-DIAG",
            message_en=structured.get("summary", raw_text[:200]) if structured else raw_text[:200],
            message_da=structured.get("summary", raw_text[:200]) if structured else raw_text[:200],
            resolved=True,
            resolved_at=datetime.utcnow(),
            ai_analysis=log_payload,
        )
        db.session.add(ai_event)
        db.session.commit()

        return jsonify({
            "status": "ok",
            "turbine_id": turbine_id,
            "structured": structured,
            "raw": raw_text,
            "event_id": ai_event.id,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ─── Helpers ──────────────────────────────────────────────────────────────────
def _reading_dict(r):
    if not r:
        return None
    return {
        "ts": r.timestamp.isoformat(),
        "power_kw": r.power_output_kw,
        "wind_ms": r.wind_speed_ms,
        "wind_dir": r.wind_direction_deg,
        "rpm": r.rotor_rpm,
        "pitch": r.pitch_angle_deg,
        "nacelle_temp": r.nacelle_temp_c,
        "bearing_temp": r.bearing_temp_c,
        "gearbox_temp": r.gearbox_temp_c,
        "generator_temp": r.generator_temp_c,
        "oil_temp": r.oil_temp_c,
        "ambient_temp": r.ambient_temp_c,
        "humidity": r.nacelle_humidity_pct,
        "vib_x": r.vibration_gearbox_x,
        "vib_y": r.vibration_gearbox_y,
        "vib_z": r.vibration_gearbox_z,
        "vib_bearing": r.vibration_bearing,
        "acoustic_db": r.acoustic_level_db,
        "voltage": r.grid_voltage_v,
        "frequency": r.grid_frequency_hz,
        "power_factor": r.power_factor,
    }

def _event_dict(e):
    lang = session.get('lang', 'en')
    return {
        "id": e.id,
        "turbine_id": e.turbine_id,
        "ts": e.timestamp.isoformat(),
        "severity": e.severity,
        "category": e.category,
        "code": e.code,
        "message": e.message_da if lang == 'da' else e.message_en,
        "message_en": e.message_en,
        "message_da": e.message_da,
        "resolved": e.resolved,
        "resolved_at": e.resolved_at.isoformat() if e.resolved_at else None,
        "ai_analysis": e.ai_analysis,
    }

# ─── Language Toggle ──────────────────────────────────────────────────────────
@app.route('/lang/<lang>')
def set_lang(lang):
    if lang in ('en', 'da'):
        session['lang'] = lang
    return redirect(request.referrer or url_for('dashboard'))

@app.context_processor
def inject_lang():
    return {"lang": session.get('lang', 'en')}

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
