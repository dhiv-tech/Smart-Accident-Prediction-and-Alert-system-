import csv
import io
import json
import math
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from severity_model import MAX_DATASET_BYTES, inspect_dataset, model_status, predict_severity, train_model

BASE_DIR = Path(__file__).resolve().parent
DATABASE = BASE_DIR / "safedrive.db"
LOCK = threading.Lock()


@contextmanager
def connect():
    connection = sqlite3.connect(DATABASE, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def initialize():
    with connect() as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS telemetry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recorded_at TEXT NOT NULL,
                risk INTEGER NOT NULL,
                confidence INTEGER NOT NULL,
                speed REAL NOT NULL,
                acceleration REAL NOT NULL,
                lane_stability INTEGER NOT NULL,
                driver_state TEXT NOT NULL,
                latitude REAL,
                longitude REAL,
                source TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sensor_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                received_at TEXT NOT NULL,
                latitude REAL,
                longitude REAL,
                accuracy_m REAL,
                altitude_m REAL,
                heading_deg REAL,
                speed_kmh REAL,
                speed_method TEXT,
                payload_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_sensor_readings_source_id ON sensor_readings(source, id DESC);
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                severity TEXT NOT NULL,
                category TEXT NOT NULL,
                message TEXT NOT NULL,
                acknowledged INTEGER NOT NULL DEFAULT 0,
                source TEXT NOT NULL DEFAULT 'manual'
            );
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                vehicle_id TEXT NOT NULL DEFAULT 'SD-001',
                incident_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                risk INTEGER NOT NULL,
                location TEXT NOT NULL,
                status INTEGER NOT NULL DEFAULT 0,
                confirmed INTEGER,
                source TEXT NOT NULL DEFAULT 'manual'
            );
            CREATE TABLE IF NOT EXISTS trips (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                distance_km REAL NOT NULL DEFAULT 0,
                peak_risk INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'active'
            );
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            INSERT OR IGNORE INTO settings(key, value) VALUES ('risk_threshold', '60');
            INSERT OR IGNORE INTO settings(key, value) VALUES ('sampling_interval', '3');
        """)
        trip_columns = {row["name"] for row in db.execute("PRAGMA table_info(trips)")}
        if "gps_fixes" not in trip_columns:
            db.execute("ALTER TABLE trips ADD COLUMN gps_fixes INTEGER NOT NULL DEFAULT 0")
        for table in ("alerts", "incidents"):
            columns = {row["name"] for row in db.execute(f"PRAGMA table_info({table})")}
            if "source" not in columns:
                db.execute(f"ALTER TABLE {table} ADD COLUMN source TEXT NOT NULL DEFAULT 'legacy'")


def json_response(handler, data, status=200):
    body = json.dumps(data).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def numeric_fields(payload, limits):
    values = {}
    for key, (minimum, maximum) in limits.items():
        value = payload.get(key)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{key} must be numeric.") from error
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(f"{key} must be between {minimum} and {maximum}.")
        values[key] = value
    return values


def observed_timestamp(value):
    if value is None:
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("observed_at must be an ISO 8601 timestamp.") from error
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat(timespec="milliseconds")


def insert_sensor_reading(source, payload):
    observed_at = observed_timestamp(payload.get("observed_at"))
    received_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    latitude = longitude = accuracy = altitude = heading = speed_kmh = None
    speed_method = None
    if source == "gps":
        coordinates = numeric_fields(payload, {
            "latitude": (-90, 90), "longitude": (-180, 180), "accuracy_m": (0, 10000),
            "altitude_m": (-1000, 100000), "heading_deg": (0, 360), "speed_mps": (0, 100),
        })
        if "latitude" not in coordinates or "longitude" not in coordinates:
            raise ValueError("GPS readings require latitude and longitude.")
        latitude, longitude = coordinates["latitude"], coordinates["longitude"]
        accuracy = coordinates.get("accuracy_m")
        altitude = coordinates.get("altitude_m")
        heading = coordinates.get("heading_deg")
        if "speed_mps" in coordinates:
            speed_kmh = coordinates["speed_mps"] * 3.6
            speed_method = "gps_receiver"
        elif accuracy is not None and accuracy <= 50:
            with connect() as db:
                previous = db.execute("SELECT latitude, longitude, accuracy_m, observed_at FROM sensor_readings WHERE source='gps' AND latitude IS NOT NULL ORDER BY datetime(observed_at) DESC, id DESC LIMIT 1").fetchone()
            if previous and (previous["accuracy_m"] is None or previous["accuracy_m"] <= 50):
                previous_time = datetime.fromisoformat(previous["observed_at"])
                elapsed = (datetime.fromisoformat(observed_at) - previous_time).total_seconds()
                if 0.5 <= elapsed <= 30:
                    lat1, lat2 = math.radians(previous["latitude"]), math.radians(latitude)
                    delta_lat = lat2 - lat1
                    delta_lon = math.radians(longitude - previous["longitude"])
                    haversine = math.sin(delta_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
                    distance_m = 6371000 * 2 * math.asin(math.sqrt(min(1, haversine)))
                    derived_speed = distance_m / elapsed
                    if derived_speed <= 70:
                        speed_kmh = derived_speed * 3.6
                        speed_method = "gps_position_delta"
    else:
        allowed = {
            "motion": {
                "acceleration_x_g": (-32, 32), "acceleration_y_g": (-32, 32), "acceleration_z_g": (-32, 32),
                "acceleration_including_gravity_x_g": (-32, 32), "acceleration_including_gravity_y_g": (-32, 32), "acceleration_including_gravity_z_g": (-32, 32),
                "rotation_rate_x_dps": (-4000, 4000), "rotation_rate_y_dps": (-4000, 4000), "rotation_rate_z_dps": (-4000, 4000),
            },
            "obd": {
                "vehicle_speed_kmh": (0, 300), "engine_rpm": (0, 20000), "throttle_pct": (0, 100), "brake_pct": (0, 100),
            },
            "vision": {
                "drowsiness_pct": (0, 100), "lane_offset_m": (-10, 10), "following_distance_m": (0, 300),
            },
        }[source]
        numeric_fields(payload, allowed)
        if source == "obd" and payload.get("seatbelt_fastened") is not None and not isinstance(payload["seatbelt_fastened"], bool):
            raise ValueError("seatbelt_fastened must be a boolean.")
        if not any(key in payload for key in allowed) and not (source == "obd" and "seatbelt_fastened" in payload):
            raise ValueError(f"{source} readings must include at least one supported sensor value.")
    payload_json = json.dumps(payload, allow_nan=False)
    with LOCK, connect() as db:
        cursor = db.execute("""INSERT INTO sensor_readings
            (source, observed_at, received_at, latitude, longitude, accuracy_m, altitude_m, heading_deg, speed_kmh, speed_method, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (source, observed_at, received_at, latitude, longitude, accuracy, altitude, heading, speed_kmh, speed_method, payload_json))
        row = db.execute("SELECT * FROM sensor_readings WHERE id=?", (cursor.lastrowid,)).fetchone()
    result = dict(row)
    result["payload"] = json.loads(result.pop("payload_json"))
    return result


def trip_gps_summary(db, started_at, ended_at):
    fixes = db.execute("""SELECT latitude, longitude, observed_at FROM sensor_readings
        WHERE source='gps' AND accuracy_m <= 50
        AND datetime(observed_at) >= datetime(?) AND datetime(observed_at) <= datetime(?)
        ORDER BY datetime(observed_at), id""", (started_at, ended_at)).fetchall()
    distance_m = 0.0
    previous = None
    for fix in fixes:
        if previous is not None:
            elapsed = (datetime.fromisoformat(fix["observed_at"]) - datetime.fromisoformat(previous["observed_at"])).total_seconds()
            lat1 = math.radians(previous["latitude"])
            lat2 = math.radians(fix["latitude"])
            delta_lat = lat2 - lat1
            delta_lon = math.radians(fix["longitude"] - previous["longitude"])
            haversine = math.sin(delta_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
            segment_m = 6371000 * 2 * math.asin(math.sqrt(min(1, haversine)))
            if 0.5 <= elapsed <= 60 and segment_m <= elapsed * 70:
                distance_m += segment_m
        previous = fix
    return round(distance_m / 1000, 3), len(fixes)


def create_telemetry(source, payload):
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    values = payload
    with LOCK, connect() as db:
        db.execute("""INSERT INTO telemetry
            (recorded_at, risk, confidence, speed, acceleration, lane_stability,
             driver_state, latitude, longitude, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (now, values["risk"], values["confidence"], values["speed"],
             values["acceleration"], values["lane_stability"], values["driver_state"],
             values.get("latitude"), values.get("longitude"), source))
        threshold = int(db.execute("SELECT value FROM settings WHERE key='risk_threshold'").fetchone()[0])
        recent = db.execute("SELECT 1 FROM alerts WHERE category='Risk threshold' AND acknowledged=0 AND datetime(created_at) >= datetime('now', '-5 minutes') LIMIT 1").fetchone()
        if values["risk"] >= threshold and not recent:
            db.execute("INSERT INTO alerts(created_at, severity, category, message, source) VALUES (?, 'high', 'Risk threshold', ?, 'model')",
                       (now, f"Predicted accident risk reached {values['risk']} (threshold {threshold})."))
            severity = "Critical" if values["risk"] >= 85 else "High"
            db.execute("INSERT INTO incidents(created_at, incident_type, severity, risk, location, source) VALUES (?, 'Predicted risk advisory', ?, ?, ?, 'model')",
                       (now, severity, values["risk"], f"{values.get('latitude') or 0:.4f}, {values.get('longitude') or 0:.4f}"))
        row = db.execute("SELECT * FROM telemetry ORDER BY id DESC LIMIT 1").fetchone()
        return dict(row)


def validate_telemetry(payload):
    required = ("risk", "confidence", "speed", "acceleration", "lane_stability", "driver_state")
    if not isinstance(payload, dict) or any(key not in payload for key in required):
        raise ValueError("Required fields: risk, confidence, speed, acceleration, lane_stability, driver_state")
    try:
        result = {
            "risk": int(payload["risk"]),
            "confidence": int(payload["confidence"]),
            "speed": float(payload["speed"]),
            "acceleration": float(payload["acceleration"]),
            "lane_stability": int(payload["lane_stability"]),
            "driver_state": str(payload["driver_state"])[:40],
            "latitude": float(payload["latitude"]) if payload.get("latitude") is not None else None,
            "longitude": float(payload["longitude"]) if payload.get("longitude") is not None else None,
        }
    except (TypeError, ValueError) as error:
        raise ValueError("Telemetry values must be numeric where applicable.") from error
    if not 0 <= result["risk"] <= 100 or not 0 <= result["confidence"] <= 100 or not 0 <= result["lane_stability"] <= 100 or result["speed"] < 0:
        raise ValueError("Risk, confidence, and lane stability must be 0-100; speed cannot be negative.")
    return result


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BASE_DIR), **kwargs)

    def log_message(self, format, *args):
        print(f"{self.log_date_time_string()} {format % args}")

    def do_GET(self):
        parsed = urlparse(self.path)
        route = parsed.path
        if route == "/api/model/status":
            return json_response(self, model_status())
        if route == "/api/health":
            with connect() as db:
                model_samples = db.execute("SELECT COUNT(*) FROM telemetry WHERE source='model'").fetchone()[0]
                sensor_count = db.execute("SELECT COUNT(*) FROM sensor_readings").fetchone()[0]
                open_incidents = db.execute("SELECT COUNT(*) FROM incidents WHERE status < 3").fetchone()[0]
            return json_response(self, {"status": "online", "database": "SQLite", "model_samples": model_samples, "sensor_readings": sensor_count, "open_incidents": open_incidents})
        if route == "/api/sensors/latest":
            latest = {}
            with connect() as db:
                for source in ("gps", "motion", "obd", "vision"):
                    row = db.execute("SELECT * FROM sensor_readings WHERE source=? ORDER BY datetime(observed_at) DESC, id DESC LIMIT 1", (source,)).fetchone()
                    if row:
                        latest[source] = dict(row)
                        latest[source]["payload"] = json.loads(latest[source].pop("payload_json"))
                    else:
                        latest[source] = None
            return json_response(self, latest)
        if route == "/api/sensors/history":
            limit = max(1, min(200, int(parse_qs(parsed.query).get("limit", ["100"])[0])))
            with connect() as db:
                rows = [dict(row) for row in db.execute("SELECT * FROM sensor_readings ORDER BY datetime(observed_at) DESC, id DESC LIMIT ?", (limit,))]
            rows.reverse()
            for row in rows:
                row["payload"] = json.loads(row.pop("payload_json"))
            return json_response(self, rows)
        if route == "/api/telemetry":
            query = parse_qs(parsed.query)
            if query.get("source") == ["device"]:
                return json_response(self, {"error": "Use /api/sensors/latest for device sensor state."}, 410)
            with connect() as db:
                latest = db.execute("SELECT * FROM telemetry WHERE source='model' ORDER BY id DESC LIMIT 1").fetchone()
                trend = [dict(row) for row in db.execute("SELECT recorded_at, risk FROM telemetry WHERE source='model' ORDER BY id DESC LIMIT 40")]
                trend.reverse()
            return json_response(self, {"latest": dict(latest) if latest else None, "trend": trend})
        if route == "/api/telemetry/history":
            limit = max(1, min(100, int(parse_qs(parsed.query).get("limit", ["40"])[0])))
            with connect() as db:
                rows = [dict(row) for row in db.execute("SELECT * FROM telemetry WHERE source='model' ORDER BY id DESC LIMIT ?", (limit,))]
            rows.reverse()
            return json_response(self, rows)
        if route == "/api/alerts":
            with connect() as db:
                rows = [dict(row) for row in db.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT 100")]
            return json_response(self, rows)
        if route == "/api/incidents":
            with connect() as db:
                rows = [dict(row) for row in db.execute("SELECT * FROM incidents ORDER BY id DESC LIMIT 100")]
            return json_response(self, rows)
        if route == "/api/trips":
            with connect() as db:
                rows = [dict(row) for row in db.execute("SELECT * FROM trips ORDER BY id DESC LIMIT 100")]
            return json_response(self, rows)
        if route == "/api/settings":
            with connect() as db:
                settings = {row["key"]: int(row["value"]) for row in db.execute("SELECT key, value FROM settings")}
            return json_response(self, settings)
        if route == "/api/analytics":
            with connect() as db:
                stats = db.execute("SELECT COUNT(*) AS samples, ROUND(AVG(risk), 1) AS average_risk, MAX(risk) AS peak_risk, ROUND(AVG(speed), 1) AS average_speed FROM telemetry WHERE source='model' AND datetime(recorded_at) >= datetime('now', '-24 hours')").fetchone()
                alerts = db.execute("SELECT COUNT(*) FROM alerts WHERE datetime(created_at) >= datetime('now', '-24 hours')").fetchone()[0]
            return json_response(self, {**dict(stats), "alerts_24h": alerts})
        if route == "/api/export.csv":
            with connect() as db:
                rows = db.execute("SELECT recorded_at, risk, confidence, speed, acceleration, lane_stability, driver_state, latitude, longitude, source FROM telemetry WHERE source='model' ORDER BY id").fetchall()
                sensor_rows = db.execute("SELECT source, observed_at, received_at, latitude, longitude, accuracy_m, altitude_m, heading_deg, speed_kmh, speed_method, payload_json FROM sensor_readings ORDER BY id").fetchall()
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(["record_type", "source", "observed_at", "received_at", "risk", "confidence", "speed_kmh", "speed_method", "acceleration_g", "lane_stability", "driver_state", "latitude", "longitude", "accuracy_m", "altitude_m", "heading_deg", "raw_payload"])
            for row in rows:
                writer.writerow(["telemetry", row["source"], row["recorded_at"], row["recorded_at"], row["risk"], row["confidence"], row["speed"], "legacy_telemetry", row["acceleration"], row["lane_stability"], row["driver_state"], row["latitude"], row["longitude"], "", "", "", ""])
            for row in sensor_rows:
                writer.writerow(["sensor_reading", row["source"], row["observed_at"], row["received_at"], "", "", row["speed_kmh"], row["speed_method"], "", "", "", row["latitude"], row["longitude"], row["accuracy_m"], row["altitude_m"], row["heading_deg"], row["payload_json"]])
            body = output.getvalue().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition", "attachment; filename=safedrive-telemetry.csv")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            return self.wfile.write(body)
        if route.startswith("/api/"):
            return json_response(self, {"error": "Not found"}, 404)
        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        route = parsed.path
        try:
            if route in ("/api/dataset/inspect", "/api/model/train"):
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError as error:
                    return json_response(self, {"error": "A valid Content-Length header is required."}, 400)
                if length <= 0:
                    return json_response(self, {"error": "Upload a non-empty dataset file."}, 400)
                if length > MAX_DATASET_BYTES:
                    self.close_connection = True
                    return json_response(self, {"error": "Dataset files must be 20 MiB or smaller."}, 413)
                contents = self.rfile.read(length)
                if len(contents) != length:
                    return json_response(self, {"error": "The dataset upload was incomplete."}, 400)
                query = parse_qs(parsed.query, keep_blank_values=True)
                filename = Path(query.get("filename", [""])[0]).name
                if route == "/api/dataset/inspect":
                    return json_response(self, inspect_dataset(contents, filename))
                target = query.get("target", [""])[0]
                features = query.get("feature", [])
                dataset_source = query.get("dataset_source", [""])[0]
                dataset_permission = query.get("dataset_permission", [""])[0]
                return json_response(self, train_model(
                    contents,
                    filename,
                    target,
                    features,
                    dataset_source,
                    dataset_permission,
                ))

            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            if route == "/api/model/predict":
                if not isinstance(payload, dict) or "features" not in payload:
                    return json_response(self, {"error": "Provide model feature values in a features object."}, 400)
                return json_response(self, predict_severity(payload["features"]))
            if route in ("/api/sensors/gps", "/api/sensors/motion", "/api/sensors/obd", "/api/sensors/vision"):
                source = route.rsplit("/", 1)[-1]
                return json_response(self, insert_sensor_reading(source, payload), 201)
            if route == "/api/telemetry":
                if payload.get("source") != "model":
                    return json_response(self, {"error": "This endpoint accepts only explicit model outputs. Send raw device readings to /api/sensors/*; simulated telemetry is disabled."}, 410)
                values = validate_telemetry(payload)
                return json_response(self, create_telemetry("model", values), 201)
            if route == "/api/alerts/sos":
                now = datetime.now(timezone.utc).isoformat(timespec="seconds")
                with connect() as db:
                    message = str(payload.get("message", "Manual SOS activated."))[:300]
                    incident_type = str(payload.get("type", "Manual SOS"))[:80]
                    risk = max(0, min(100, int(payload.get("risk", 100))))
                    location = str(payload.get("location", "Location unavailable"))[:120]
                    source = "scenario_demo" if "simulated" in incident_type.lower() else "manual_sos"
                    db.execute("INSERT INTO alerts(created_at, severity, category, message, source) VALUES (?, 'critical', 'SOS', ?, ?)", (now, message, source))
                    db.execute("INSERT INTO incidents(created_at, incident_type, severity, risk, location, source) VALUES (?, ?, 'Critical', ?, ?, ?)", (now, incident_type, risk, location, source))
                    alert_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
                return json_response(self, {"id": alert_id, "status": "created"}, 201)
            if route == "/api/alerts/sos/cancel":
                now = datetime.now(timezone.utc).isoformat(timespec="seconds")
                message = str(payload.get("message", "SOS countdown canceled."))[:300]
                with connect() as db:
                    db.execute("INSERT INTO alerts(created_at, severity, category, message, acknowledged, source) VALUES (?, 'info', 'SOS canceled', ?, 1, 'local_operator')", (now, message))
                    alert_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
                return json_response(self, {"id": alert_id, "status": "canceled"}, 201)
            if route.startswith("/api/alerts/") and route.endswith("/ack"):
                alert_id = int(route.split("/")[-2])
                with connect() as db:
                    cursor = db.execute("UPDATE alerts SET acknowledged=1 WHERE id=?", (alert_id,))
                return json_response(self, {"updated": cursor.rowcount == 1})
            if route.startswith("/api/incidents/"):
                parts = route.strip("/").split("/")
                incident_id = int(parts[2])
                with connect() as db:
                    if len(parts) == 4 and parts[3] == "status":
                        status = int(payload.get("status", -1))
                        if status not in range(4):
                            return json_response(self, {"error": "Status must be between 0 and 3."}, 400)
                        cursor = db.execute("UPDATE incidents SET status=? WHERE id=?", (status, incident_id))
                    elif len(parts) == 4 and parts[3] == "feedback":
                        confirmed = payload.get("confirmed")
                        if not isinstance(confirmed, bool):
                            return json_response(self, {"error": "confirmed must be a boolean."}, 400)
                        cursor = db.execute("UPDATE incidents SET confirmed=? WHERE id=?", (int(confirmed), incident_id))
                    else:
                        return json_response(self, {"error": "Not found"}, 404)
                return json_response(self, {"updated": cursor.rowcount == 1})
            if route == "/api/trips/start":
                now = datetime.now(timezone.utc).isoformat(timespec="seconds")
                with connect() as db:
                    active = db.execute("SELECT id FROM trips WHERE status='active'").fetchone()
                    if active:
                        return json_response(self, {"error": "A trip is already active."}, 409)
                    cursor = db.execute("INSERT INTO trips(started_at, status) VALUES (?, 'active')", (now,))
                return json_response(self, {"id": cursor.lastrowid, "status": "active"}, 201)
            if route == "/api/trips/stop":
                now = datetime.now(timezone.utc).isoformat(timespec="seconds")
                with connect() as db:
                    active = db.execute("SELECT id, started_at FROM trips WHERE status='active' ORDER BY id DESC LIMIT 1").fetchone()
                    if not active:
                        return json_response(self, {"error": "There is no active trip."}, 409)
                    distance, gps_fixes = trip_gps_summary(db, active["started_at"], now)
                    db.execute("UPDATE trips SET ended_at=?, distance_km=?, gps_fixes=?, status='completed' WHERE id=?", (now, distance, gps_fixes, active["id"]))
                return json_response(self, {"status": "completed"})
            if route == "/api/settings":
                threshold = int(payload.get("risk_threshold", 60))
                interval = int(payload.get("sampling_interval", 3))
                if not 20 <= threshold <= 95 or not 2 <= interval <= 60:
                    return json_response(self, {"error": "Threshold must be 20-95 and sampling interval 2-60 seconds."}, 400)
                with connect() as db:
                    db.executemany("INSERT INTO settings(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", [("risk_threshold", str(threshold)), ("sampling_interval", str(interval))])
                return json_response(self, {"risk_threshold": threshold, "sampling_interval": interval})
            return json_response(self, {"error": "Not found"}, 404)
        except (ValueError, json.JSONDecodeError) as error:
            return json_response(self, {"error": str(error)}, 400)


def main():
    initialize()
    host = os.environ.get("SAFEDRIVE_HOST", "127.0.0.1")
    port = int(os.environ.get("SAFEDRIVE_PORT", "8000"))
    server = ThreadingHTTPServer((host, port), Handler)
    display_host = "localhost" if host == "127.0.0.1" else host
    print(f"SafeDrive monitoring server: http://{display_host}:{port}")
    if host == "0.0.0.0":
        print("WARNING: unauthenticated network mode; use only on a trusted private network.")
    print(f"SQLite database: {DATABASE}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down SafeDrive.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
