"""
Wind Monitor - Database Models
SQLite via SQLAlchemy
"""
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()


class Turbine(db.Model):
    __tablename__ = 'turbines'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), nullable=False)
    make = db.Column(db.String(64), nullable=False)
    model = db.Column(db.String(64), nullable=False)
    serial_number = db.Column(db.String(64), unique=True, nullable=False)
    rated_power_kw = db.Column(db.Float, nullable=False)
    hub_height_m = db.Column(db.Float, nullable=False)
    rotor_diameter_m = db.Column(db.Float, nullable=False)
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    build_date = db.Column(db.Date, nullable=False)
    last_service_date = db.Column(db.Date, nullable=False)
    status = db.Column(db.String(16), default='operational')  # operational, warning, fault, offline
    notes = db.Column(db.Text, default='')

    readings = db.relationship('SensorReading', backref='turbine', lazy='dynamic')
    events = db.relationship('TurbineEvent', backref='turbine', lazy='dynamic')


class SensorReading(db.Model):
    __tablename__ = 'sensor_readings'
    id = db.Column(db.Integer, primary_key=True)
    turbine_id = db.Column(db.Integer, db.ForeignKey('turbines.id'), nullable=False, index=True)
    timestamp = db.Column(db.DateTime, nullable=False, index=True)
    resolution = db.Column(db.String(8), default='min')  # 'min' or 'hour'

    # Power & Generation
    power_output_kw = db.Column(db.Float)
    power_factor = db.Column(db.Float)
    grid_voltage_v = db.Column(db.Float)
    grid_frequency_hz = db.Column(db.Float)
    current_a = db.Column(db.Float)

    # Wind
    wind_speed_ms = db.Column(db.Float)
    wind_direction_deg = db.Column(db.Float)
    pitch_angle_deg = db.Column(db.Float)
    rotor_rpm = db.Column(db.Float)

    # Temperature (°C)
    nacelle_temp_c = db.Column(db.Float)
    bearing_temp_c = db.Column(db.Float)
    gearbox_temp_c = db.Column(db.Float)
    generator_temp_c = db.Column(db.Float)
    oil_temp_c = db.Column(db.Float)
    ambient_temp_c = db.Column(db.Float)

    # Humidity
    nacelle_humidity_pct = db.Column(db.Float)

    # Vibration (g)
    vibration_gearbox_x = db.Column(db.Float)
    vibration_gearbox_y = db.Column(db.Float)
    vibration_gearbox_z = db.Column(db.Float)
    vibration_bearing = db.Column(db.Float)
    vibration_blade_1 = db.Column(db.Float)
    vibration_blade_2 = db.Column(db.Float)
    vibration_blade_3 = db.Column(db.Float)

    # Acoustic
    acoustic_level_db = db.Column(db.Float)


class TurbineEvent(db.Model):
    __tablename__ = 'turbine_events'
    id = db.Column(db.Integer, primary_key=True)
    turbine_id = db.Column(db.Integer, db.ForeignKey('turbines.id'), nullable=False, index=True)
    timestamp = db.Column(db.DateTime, nullable=False, index=True)
    severity = db.Column(db.String(16), nullable=False)  # info, warning, fault, critical
    category = db.Column(db.String(32), nullable=False)  # vibration, temperature, power, communication, maintenance
    code = db.Column(db.String(16), nullable=False)
    message_en = db.Column(db.Text, nullable=False)
    message_da = db.Column(db.Text, nullable=False)
    resolved = db.Column(db.Boolean, default=False)
    resolved_at = db.Column(db.DateTime, nullable=True)
    ai_analysis = db.Column(db.Text, nullable=True)  # Placeholder for AI-generated analysis


class MLInsight(db.Model):
    """Stores the latest ML analysis result for each turbine (one row per turbine, upserted)."""
    __tablename__ = 'ml_insights'
    id            = db.Column(db.Integer, primary_key=True)
    turbine_id    = db.Column(db.Integer, db.ForeignKey('turbines.id'),
                              nullable=False, index=True, unique=True)
    computed_at   = db.Column(db.DateTime, nullable=False)
    health_score  = db.Column(db.Float, nullable=True)    # 0-100; None = insufficient data
    anomaly_scores = db.Column(db.Text, nullable=True)    # JSON: {metric: z_score}
    trend_data     = db.Column(db.Text, nullable=True)    # JSON: {metric: {slope_per_hour, direction}}
    peer_ranks     = db.Column(db.Text, nullable=True)    # JSON: {metric: percentile_int 0-100}
    ml_summary     = db.Column(db.Text, nullable=True)    # Plain text block injected into Claude prompt
    raw_results    = db.Column(db.Text, nullable=True)    # Full JSON debug blob
