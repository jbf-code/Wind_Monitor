"""
Wind Monitor - Seed Script
Generates 3 weeks of realistic dummy data for 16 turbines in western Jutland, Denmark.
Strategy:
  - Weeks 1-2: hourly aggregates (resolution='hour') → ~5,376 rows
  - Week 3 (most recent): per-minute data (resolution='min') → ~161,280 rows
  - Total: ~167K rows — fast to generate, Railway-friendly
"""
import random
import math
from datetime import datetime, timedelta, date
from database import db, Turbine, SensorReading, TurbineEvent

# ──────────────────────────────────────────────
# TURBINE FLEET DEFINITION
# 16 turbines in western Jutland (Esbjerg / Ringkøbing region)
# ──────────────────────────────────────────────
TURBINE_FLEET = [
    # id, name, make, model, serial, rated_kw, hub_h, rotor_d, lat, lon, build_year, last_service
    (1,  "Ørn-01", "Vestas",          "V150-4.5",        "VES-V150-001-DK", 4500, 105, 150, 55.8723, 8.2841, 2019, "2024-03-15"),
    (2,  "Ørn-02", "Vestas",          "V150-4.5",        "VES-V150-002-DK", 4500, 105, 150, 55.8751, 8.2903, 2019, "2024-03-15"),
    (3,  "Ørn-03", "Vestas",          "V150-4.5",        "VES-V150-003-DK", 4500, 105, 150, 55.8699, 8.2967, 2020, "2024-04-20"),
    (4,  "Ørn-04", "Vestas",          "V150-4.5",        "VES-V150-004-DK", 4500, 105, 150, 55.8672, 8.2889, 2020, "2024-04-20"),
    (5,  "Falk-01","Siemens Gamesa",  "SG 5.0-145",      "SG-5145-001-DK",  5000, 110, 145, 55.8810, 8.3102, 2021, "2024-05-10"),
    (6,  "Falk-02","Siemens Gamesa",  "SG 5.0-145",      "SG-5145-002-DK",  5000, 110, 145, 55.8839, 8.3168, 2021, "2024-05-10"),
    (7,  "Falk-03","Siemens Gamesa",  "SG 5.0-145",      "SG-5145-003-DK",  5000, 110, 145, 55.8867, 8.3089, 2021, "2024-06-01"),
    (8,  "Falk-04","Siemens Gamesa",  "SG 5.0-145",      "SG-5145-004-DK",  5000, 110, 145, 55.8892, 8.3022, 2022, "2024-06-01"),
    (9,  "Vind-01","GE Renewable",    "GE 4.8-158",      "GE-48158-001-DK", 4800, 120, 158, 55.8615, 8.3215, 2020, "2024-02-28"),
    (10, "Vind-02","GE Renewable",    "GE 4.8-158",      "GE-48158-002-DK", 4800, 120, 158, 55.8588, 8.3281, 2020, "2024-02-28"),
    (11, "Vind-03","GE Renewable",    "GE 4.8-158",      "GE-48158-003-DK", 4800, 120, 158, 55.8561, 8.3197, 2021, "2024-07-12"),
    (12, "Vind-04","GE Renewable",    "GE 4.8-158",      "GE-48158-004-DK", 4800, 120, 158, 55.8534, 8.3134, 2021, "2024-07-12"),
    (13, "Stork-01","Vestas",         "V136-4.2",        "VES-V136-001-DK", 4200,  82, 136, 55.8920, 8.2745, 2017, "2023-11-05"),
    (14, "Stork-02","Vestas",         "V136-4.2",        "VES-V136-002-DK", 4200,  82, 136, 55.8948, 8.2812, 2017, "2023-11-05"),
    (15, "Stork-03","Vestas",         "V136-4.2",        "VES-V136-003-DK", 4200,  82, 136, 55.8975, 8.2734, 2018, "2024-01-18"),
    (16, "Stork-04","Vestas",         "V136-4.2",        "VES-V136-004-DK", 4200,  82, 136, 55.8998, 8.2668, 2018, "2024-01-18"),
]

# Turbines with known issues for realistic scenarios
PROBLEM_TURBINES = {
    3:  {"issue": "bearing_overheat",  "severity": "warning"},
    7:  {"issue": "gearbox_vibration", "severity": "fault"},
    13: {"issue": "offline",           "severity": "critical"},
    10: {"issue": "performance_drop",  "severity": "warning"},
}

# ──────────────────────────────────────────────
# PHYSICS HELPERS
# ──────────────────────────────────────────────
def wind_to_power(wind_ms, rated_kw, rated_wind=12.0, cut_in=3.0, cut_out=25.0):
    """Simplified power curve calculation."""
    if wind_ms < cut_in or wind_ms > cut_out:
        return 0.0
    if wind_ms >= rated_wind:
        return rated_kw * random.uniform(0.95, 1.0)
    # Cubic region
    frac = ((wind_ms - cut_in) / (rated_wind - cut_in)) ** 3
    return rated_kw * frac * random.uniform(0.92, 1.0)

def wind_speed_at_time(base_hour, day_of_year=0):
    """Realistic wind speed pattern (m/s) for western Jutland.
    Uses day_of_year to create a slow multi-day wind cycle so data
    looks realistic across the full 3-week range with no seams.
    """
    # Daily variation (peaks mid-afternoon, dips early morning)
    daily = 2.0 * math.sin((base_hour % 24 - 14) * math.pi / 12)
    # Slow synoptic cycle: ~4-day weather system period
    synoptic = 3.0 * math.sin(day_of_year * 2 * math.pi / 4.3)
    base = 8.5 + daily + synoptic + random.gauss(0, 0.8)
    return max(0.5, min(28.0, base))

def add_noise(val, pct=0.02):
    return val * (1 + random.gauss(0, pct))

# ──────────────────────────────────────────────
# READING GENERATOR
# ──────────────────────────────────────────────
def generate_reading(turbine_id, rated_kw, ts, hour_of_day, issue=None, resolution='min', day_offset=0):
    wind = wind_speed_at_time(hour_of_day, day_of_year=day_offset)
    power = wind_to_power(wind, rated_kw)

    # Apply issue overrides
    bearing_temp_extra = 0
    gearbox_vib_extra = 0
    if issue == "bearing_overheat":
        bearing_temp_extra = random.uniform(15, 30)
    elif issue == "gearbox_vibration":
        gearbox_vib_extra = random.uniform(0.3, 0.8)
    elif issue == "offline":
        power = 0
        wind = add_noise(wind, 0.01)
    elif issue == "performance_drop":
        power *= random.uniform(0.6, 0.75)

    rpm = max(0, (power / rated_kw) * 15.0 + random.gauss(0, 0.3)) if power > 0 else 0
    pitch = max(-2, min(90, 15 - wind * 1.2 + random.gauss(0, 0.5))) if power > 0 else 90

    outdoor_temp = random.gauss(5, 2)  # Danish Feb avg ~5°C (used for mechanical temps)
    # Turbine housing temp: electronics + heat dissipation in enclosed box.
    # Baseline ~30°C at idle, rises ~10°C under full load, with small noise.
    load_frac = min(1.0, power / rated_kw) if power > 0 else 0.0
    housing_temp = 30.0 + load_frac * 10.0 + random.gauss(0, 2.5)
    housing_temp = round(max(18.0, min(68.0, housing_temp)), 1)

    return SensorReading(
        turbine_id=turbine_id,
        timestamp=ts,
        resolution=resolution,

        power_output_kw=round(max(0, power + random.gauss(0, 10)), 2),
        power_factor=round(random.uniform(0.95, 0.99), 3),
        grid_voltage_v=round(random.gauss(10500, 50), 1),
        grid_frequency_hz=round(random.gauss(50.0, 0.02), 3),
        current_a=round(max(0, power / 10.5 + random.gauss(0, 2)), 2),

        wind_speed_ms=round(wind, 2),
        wind_direction_deg=round(random.gauss(220, 30) % 360, 1),
        pitch_angle_deg=round(pitch, 2),
        rotor_rpm=round(rpm, 2),

        nacelle_temp_c=round(outdoor_temp + random.uniform(5, 15), 1),
        bearing_temp_c=round(outdoor_temp + 30 + bearing_temp_extra + random.gauss(0, 1.5), 1),
        gearbox_temp_c=round(outdoor_temp + 40 + random.gauss(0, 2), 1),
        generator_temp_c=round(outdoor_temp + 45 + random.gauss(0, 2), 1),
        oil_temp_c=round(outdoor_temp + 35 + random.gauss(0, 1.5), 1),
        ambient_temp_c=housing_temp,
        nacelle_humidity_pct=round(random.uniform(30, 70), 1),

        vibration_gearbox_x=round(max(0, 0.05 + gearbox_vib_extra + random.gauss(0, 0.01)), 4),
        vibration_gearbox_y=round(max(0, 0.04 + gearbox_vib_extra + random.gauss(0, 0.01)), 4),
        vibration_gearbox_z=round(max(0, 0.03 + gearbox_vib_extra * 0.8 + random.gauss(0, 0.008)), 4),
        vibration_bearing=round(max(0, 0.02 + random.gauss(0, 0.005)), 4),
        vibration_blade_1=round(max(0, 0.015 + random.gauss(0, 0.003)), 4),
        vibration_blade_2=round(max(0, 0.014 + random.gauss(0, 0.003)), 4),
        vibration_blade_3=round(max(0, 0.016 + random.gauss(0, 0.003)), 4),

        acoustic_level_db=round(max(30, 65 + (power / rated_kw) * 15 + random.gauss(0, 2)), 1),
    )

# ──────────────────────────────────────────────
# EVENTS / ALERTS
# ──────────────────────────────────────────────
EVENTS_TEMPLATE = [
    {
        "tid": 3, "severity": "warning", "category": "temperature",
        "code": "TEMP-001",
        "msg_en": "Main bearing temperature elevated: 78.3°C (threshold: 70°C). Trend increasing over past 6 hours. Recommend inspection within 72 hours.",
        "msg_da": "Hovedlejetemperatur forhøjet: 78,3°C (grænse: 70°C). Stigende tendens over de seneste 6 timer. Anbefalet inspektion inden for 72 timer.",
        "resolved": False, "days_ago": 2, "ai": "AI analysis (Claude): Bearing temperature trend suggests early-stage lubrication degradation. P-F curve estimate: 18–25 days to functional failure without intervention. Priority: Schedule maintenance within 72 hours."
    },
    {
        "tid": 7, "severity": "fault", "category": "vibration",
        "code": "VIB-002",
        "msg_en": "Gearbox vibration anomaly detected on X-axis: 0.82g (threshold: 0.5g). Possible gear tooth wear. Turbine operating at reduced capacity (60%).",
        "msg_da": "Gearkassevibration anomali registreret på X-aksen: 0,82g (grænse: 0,5g). Mulig tandhjulsslitage. Møllen kører med reduceret kapacitet (60%).",
        "resolved": False, "days_ago": 1, "ai": "AI analysis (Claude): Vibration signature matches early-stage gear tooth micro-pitting. Recommend acoustic emission test. Estimated remaining useful life: 30–60 days at current load."
    },
    {
        "tid": 13, "severity": "critical", "category": "communication",
        "code": "COMM-003",
        "msg_en": "Turbine offline. No data received for 14 hours. Last known state: shutdown due to grid fault. Remote restart attempted × 3, failed. On-site inspection required.",
        "msg_da": "Mølle offline. Ingen data modtaget i 14 timer. Sidst kendte tilstand: nedlukning pga. netfejl. Fjerngenstartsforsøg × 3 mislykkedes. Inspektion på stedet påkrævet.",
        "resolved": False, "days_ago": 0, "ai": "AI analysis (Claude): Communication loss following grid fault event. SCADA logs show controlled shutdown sequence completed. Physical inspection needed to confirm safe state before restart."
    },
    {
        "tid": 10, "severity": "warning", "category": "performance",
        "code": "PERF-004",
        "msg_en": "Performance degradation detected. Actual output 31% below power curve prediction for current wind conditions. Possible blade soiling or pitch calibration drift.",
        "msg_da": "Ydelsesforringelse registreret. Faktisk produktion 31% under effektkurveforudsigelse ved aktuelle vindforhold. Mulig blade-tilsmudsning eller pitchkalibreringsdrift.",
        "resolved": False, "days_ago": 3, "ai": "AI analysis (Claude): Power coefficient (Cp) analysis shows consistent underperformance vs. fleet average. Blade soiling probability: 73%. Recommend drone inspection and cleaning. Estimated annual loss at current rate: 127 MWh."
    },
    {
        "tid": 1, "severity": "info", "category": "maintenance",
        "code": "MAINT-005",
        "msg_en": "Scheduled maintenance completed. Gearbox oil replaced, blade leading edges inspected and sealed. All systems nominal.",
        "msg_da": "Planlagt vedligeholdelse gennemført. Gearkasseolie udskiftet, bladforkanter inspiceret og forseglet. Alle systemer nominelle.",
        "resolved": True, "days_ago": 8, "ai": None
    },
    {
        "tid": 5, "severity": "warning", "category": "vibration",
        "code": "VIB-006",
        "msg_en": "Transient vibration spike on blade 2 (0.38g). Duration: 4 minutes. Possible bird strike or blade icing event. Monitoring increased frequency.",
        "msg_da": "Forbigående vibrationstop på blad 2 (0,38g). Varighed: 4 minutter. Mulig fugleslag eller isningshændelse på blad. Overvågningsfrekvens øget.",
        "resolved": True, "days_ago": 5, "ai": "AI analysis (Claude): Vibration pattern does not match structural damage signature. Most likely transient aerodynamic event (icing or debris). Monitoring for recurrence."
    },
    {
        "tid": 9, "severity": "info", "category": "performance",
        "code": "PERF-007",
        "msg_en": "Turbine performing above average. Power output 4.2% above fleet median for equivalent wind conditions.",
        "msg_da": "Mølle præsterer over gennemsnittet. Elproduktion 4,2% over flådemedianen ved tilsvarende vindforhold.",
        "resolved": True, "days_ago": 6, "ai": None
    },
    {
        "tid": 15, "severity": "warning", "category": "temperature",
        "code": "TEMP-008",
        "msg_en": "Generator temperature high: 92°C. Cooling system efficiency check recommended. Operating within safe limits but trending upward.",
        "msg_da": "Generatortemperatur høj: 92°C. Anbefalet kontrol af kølesystemeffektivitet. Opererer inden for sikre grænser, men stigende tendens.",
        "resolved": False, "days_ago": 4, "ai": "AI analysis (Claude): Generator cooling performance degraded by estimated 12%. Likely cause: filter clogging or cooling fluid level. Low urgency — schedule inspection at next planned access."
    },
]


def seed_database(app):
    """Seed all turbines, sensor data, and events."""
    with app.app_context():
        if Turbine.query.count() > 0:
            print("Database already seeded — skipping.")
            return

        # Fixed seed → deterministic, identical data on every re-seed
        random.seed(42)

        now = datetime.utcnow().replace(second=0, microsecond=0)
        three_weeks_ago = now - timedelta(weeks=3)
        one_week_ago    = now - timedelta(weeks=1)

        print("Seeding turbines...")
        turbine_map = {}
        for row in TURBINE_FLEET:
            (tid, name, make, model, serial, rated_kw, hub_h, rotor_d,
             lat, lon, build_year, last_svc) = row

            issue_info = PROBLEM_TURBINES.get(tid, {})
            status = "operational"
            if issue_info.get("severity") == "warning":
                status = "warning"
            elif issue_info.get("severity") == "fault":
                status = "fault"
            elif issue_info.get("severity") == "critical":
                status = "offline"

            t = Turbine(
                id=tid, name=name, make=make, model=model, serial_number=serial,
                rated_power_kw=rated_kw, hub_height_m=hub_h, rotor_diameter_m=rotor_d,
                latitude=lat, longitude=lon,
                build_date=date(build_year, random.randint(1, 12), random.randint(1, 28)),
                last_service_date=datetime.strptime(last_svc, "%Y-%m-%d").date(),
                status=status,
            )
            db.session.add(t)
            turbine_map[tid] = (rated_kw, issue_info.get("issue"))

        db.session.commit()
        print(f"  → {len(TURBINE_FLEET)} turbines created")

        # ── HOURLY DATA: weeks 1 & 2 ───────────────────────
        print("Seeding hourly data (weeks 1-2)...")
        batch = []
        ts = three_weeks_ago
        while ts < one_week_ago:
            day_offset = (ts - three_weeks_ago).total_seconds() / 86400.0
            for tid, (rated_kw, issue) in turbine_map.items():
                r = generate_reading(tid, rated_kw, ts, ts.hour, issue,
                                     resolution='hour', day_offset=day_offset)
                batch.append(r)
            ts += timedelta(hours=1)
            if len(batch) >= 5000:
                db.session.bulk_save_objects(batch)
                db.session.commit()
                batch = []

        if batch:
            db.session.bulk_save_objects(batch)
            db.session.commit()
        print(f"  → Hourly data committed")

        # ── MINUTE DATA: week 3 (most recent) ──────────────
        print("Seeding minute data (week 3, most recent)...")
        batch = []
        ts = one_week_ago
        count = 0
        while ts <= now:
            day_offset = (ts - three_weeks_ago).total_seconds() / 86400.0
            for tid, (rated_kw, issue) in turbine_map.items():
                r = generate_reading(tid, rated_kw, ts, ts.hour, issue,
                                     resolution='min', day_offset=day_offset)
                batch.append(r)
            ts += timedelta(minutes=1)
            count += 1
            if len(batch) >= 8000:
                db.session.bulk_save_objects(batch)
                db.session.commit()
                batch = []
                print(f"    minute {count}/{7*24*60}...")

        if batch:
            db.session.bulk_save_objects(batch)
            db.session.commit()
        print(f"  → Minute data committed")

        # ── EVENTS / ALERTS ────────────────────────────────
        print("Seeding events...")
        for evt in EVENTS_TEMPLATE:
            ts_evt = now - timedelta(days=evt["days_ago"], hours=random.randint(0, 8))
            ev = TurbineEvent(
                turbine_id=evt["tid"],
                timestamp=ts_evt,
                severity=evt["severity"],
                category=evt["category"],
                code=evt["code"],
                message_en=evt["msg_en"],
                message_da=evt["msg_da"],
                resolved=evt["resolved"],
                resolved_at=ts_evt + timedelta(hours=random.randint(2, 24)) if evt["resolved"] else None,
                ai_analysis=evt.get("ai"),
            )
            db.session.add(ev)

        db.session.commit()
        print(f"  → {len(EVENTS_TEMPLATE)} events created")
        print("Seeding complete ✓")
