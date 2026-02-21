# WindMonitor 🌬️

AI-powered wind turbine farm monitoring dashboard.

## Quick Start (Local)

```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
# Login: admin / JBF_2026
```

## Deploy to Railway

1. Push this folder to a GitHub repo
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
3. Select the repo — Railway auto-detects Python
4. Set environment variables in Railway dashboard:
   - `SECRET_KEY` → any random long string
   - `ADMIN_PASSWORD` → JBF_2026 (or change it)
   - `ANTHROPIC_API_KEY` → your Claude API key (optional, for AI features)
   - `GOOGLE_GEMINI_API_KEY` → your Gemini key (optional)
5. Deploy → the app seeds its database on first boot (~2-3 min for full 3 weeks of data)

## Architecture

```
app.py              Flask app + REST API + auth
database.py         SQLAlchemy models (Turbine, SensorReading, TurbineEvent)
seed_data.py        Generates 3 weeks of realistic dummy data
templates/
  base.html         Nav, layout, Leaflet, Chart.js CDN includes
  login.html        Login page
  dashboard.html    Fleet overview: map + KPIs + power chart + event log
  turbine_detail.html  Per-turbine: gauges + 4× time-series charts + event log
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/fleet` | All 16 turbines with latest readings |
| GET | `/api/turbine/:id` | Single turbine + events |
| GET | `/api/turbine/:id/chart?metric=X&range=24h` | Time-series data |
| GET | `/api/fleet/power` | Total fleet output (48h) |
| GET | `/api/events` | Recent events/alerts |
| POST | `/api/ai/analyze/:id` | AI diagnostic stub |
| GET | `/status` | Health check |

## Chart Metrics Available

`power_output_kw`, `wind_speed_ms`, `rotor_rpm`, `pitch_angle_deg`,
`nacelle_temp_c`, `bearing_temp_c`, `gearbox_temp_c`, `generator_temp_c`,
`oil_temp_c`, `ambient_temp_c`, `vibration_gearbox_x`, `vibration_gearbox_y`,
`vibration_gearbox_z`, `vibration_bearing`, `acoustic_level_db`,
`nacelle_humidity_pct`, `grid_voltage_v`, `grid_frequency_hz`

## AI Integration (Claude)

Set `ANTHROPIC_API_KEY` and extend `api_ai_analyze()` in `app.py`.
The endpoint already receives the turbine ID and can be extended to:
- Fetch the latest 1,000 sensor readings
- Include active alerts
- Call `anthropic.messages.create()` with a diagnostic prompt
- Return the analysis as JSON

## Data Strategy (Railway-optimised)

- **Weeks 1–2**: hourly aggregates (~5,376 rows)
- **Week 3** (most recent): per-minute data (~161,280 rows)
- **Total**: ~167K rows — seeds in ~60–120 seconds on Railway free tier

## Turbine Fleet — Western Jutland, Denmark

| Group | Make | Model | Count |
|-------|------|-------|-------|
| Ørn   | Vestas | V150-4.5 MW | 4 |
| Falk  | Siemens Gamesa | SG 5.0-145 | 4 |
| Vind  | GE Renewable | GE 4.8-158 | 4 |
| Stork | Vestas | V136-4.2 MW | 4 |

## Known Issues / Simulated Scenarios

- **Ørn-03** (T-3): Bearing overheating — Warning
- **Falk-03** (T-7): Gearbox vibration anomaly — Fault
- **Stork-01** (T-13): Offline (grid fault, comms loss) — Critical
- **Vind-02** (T-10): Performance degradation (blade soiling) — Warning
