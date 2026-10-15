"""
RoadSense AI - Database Management Layer
Provides robust, parameterized SQLite persistence for incident evidence,
detection bounding boxes, geospatial metadata, and human review status.
Guarantees zero raw SQL concatenation.
"""

import json
import math
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core import config
from core.logger import get_logger

logger = get_logger("roadsense.db")


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Create a thread-safe connection to the SQLite database."""
    target_path = db_path or config.DB_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[Path] = None) -> None:
    """Initialize database tables and performance indices."""
    schema = """
    CREATE TABLE IF NOT EXISTS incidents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        incident_uuid TEXT UNIQUE NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        image_path TEXT NOT NULL,
        thumbnail_path TEXT,
        video_source_path TEXT,
        video_frame_number INTEGER,
        
        -- Detection Metadata
        checkpoint_id TEXT NOT NULL,
        detected_classes TEXT NOT NULL, -- JSON string of array
        detection_count INTEGER NOT NULL,
        max_confidence REAL NOT NULL,
        raw_detections TEXT NOT NULL,   -- JSON string of [{class, confidence, bbox}]
        
        -- Geospatial Provenance
        latitude REAL,
        longitude REAL,
        location_provenance TEXT NOT NULL CHECK(location_provenance IN ('EXIF_GPS', 'USER_ENTERED', 'EXPLICIT_SYNTHETIC_DEMO', 'UNKNOWN')),
        road_name TEXT,
        road_classification TEXT DEFAULT 'UNKNOWN' CHECK(road_classification IN ('PRIMARY_ARTERIAL', 'SECONDARY_COLLECTOR', 'RESIDENTIAL_LOCAL', 'UNKNOWN')),
        
        -- Priority & Review
        priority_score REAL NOT NULL,
        priority_level TEXT NOT NULL CHECK(priority_level IN ('LOW', 'MODERATE', 'HIGH', 'CRITICAL', 'INSUFFICIENT_INFO')),
        priority_formula_version TEXT NOT NULL,
        
        -- Human-in-the-Loop Review
        review_status TEXT DEFAULT 'PENDING' CHECK(review_status IN ('PENDING', 'VERIFIED', 'MODIFIED', 'REJECTED')),
        human_verified_classes TEXT,
        human_urgency_modifier REAL DEFAULT 1.0,
        reviewer_notes TEXT,
        reviewed_at DATETIME
    );

    CREATE INDEX IF NOT EXISTS idx_incidents_timestamp ON incidents(timestamp);
    CREATE INDEX IF NOT EXISTS idx_incidents_priority ON incidents(priority_level);
    CREATE INDEX IF NOT EXISTS idx_incidents_location ON incidents(latitude, longitude);
    """
    with get_connection(db_path) as conn:
        conn.executescript(schema)
        conn.commit()
    logger.info("Database initialized successfully.")


def insert_incident(
    image_path: str,
    checkpoint_id: str,
    detected_classes: List[str],
    detection_count: int,
    max_confidence: float,
    raw_detections: List[Dict[str, Any]],
    priority_score: float,
    priority_level: str,
    priority_formula_version: str = config.PRIORITY_FORMULA_VERSION,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    location_provenance: str = "UNKNOWN",
    road_name: Optional[str] = None,
    road_classification: str = "UNKNOWN",
    thumbnail_path: Optional[str] = None,
    video_source_path: Optional[str] = None,
    video_frame_number: Optional[int] = None,
    db_path: Optional[Path] = None,
) -> str:
    """Insert a new incident record using strict parameterized SQL."""
    inc_uuid = f"INC-{datetime.utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"

    query = """
    INSERT INTO incidents (
        incident_uuid, image_path, thumbnail_path, video_source_path, video_frame_number,
        checkpoint_id, detected_classes, detection_count, max_confidence, raw_detections,
        latitude, longitude, location_provenance, road_name, road_classification,
        priority_score, priority_level, priority_formula_version
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    params = (
        inc_uuid,
        image_path,
        thumbnail_path,
        video_source_path,
        video_frame_number,
        checkpoint_id,
        json.dumps(detected_classes),
        detection_count,
        max_confidence,
        json.dumps(raw_detections),
        latitude,
        longitude,
        location_provenance,
        road_name,
        road_classification,
        priority_score,
        priority_level,
        priority_formula_version,
    )

    with get_connection(db_path) as conn:
        conn.execute(query, params)
        conn.commit()

    logger.info(f"Inserted incident {inc_uuid} (Priority: {priority_level}, Score: {priority_score:.1f})")
    return inc_uuid


def get_incident_by_uuid(inc_uuid: str, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Retrieve single incident record by UUID."""
    query = "SELECT * FROM incidents WHERE incident_uuid = ?"
    with get_connection(db_path) as conn:
        cursor = conn.execute(query, (inc_uuid,))
        row = cursor.fetchone()
        if not row:
            return None
        res = dict(row)
        res["detected_classes"] = json.loads(res["detected_classes"])
        res["raw_detections"] = json.loads(res["raw_detections"])
        if res.get("human_verified_classes"):
            res["human_verified_classes"] = json.loads(res["human_verified_classes"])
        return res


def get_all_incidents(
    class_filter: Optional[str] = None,
    priority_filter: Optional[str] = None,
    review_filter: Optional[str] = None,
    search_query: Optional[str] = None,
    limit: int = 200,
    offset: int = 0,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Retrieve filtered incident records using parameterized conditions."""
    conditions: List[str] = []
    params: List[Any] = []

    if class_filter and class_filter != "ALL":
        conditions.append("detected_classes LIKE ?")
        params.append(f'%"{class_filter}"%')

    if priority_filter and priority_filter != "ALL":
        conditions.append("priority_level = ?")
        params.append(priority_filter)

    if review_filter and review_filter != "ALL":
        conditions.append("review_status = ?")
        params.append(review_filter)

    if search_query:
        conditions.append("(incident_uuid LIKE ? OR road_name LIKE ? OR reviewer_notes LIKE ?)")
        wildcard = f"%{search_query}%"
        params.extend([wildcard, wildcard, wildcard])

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    query = f"""
    SELECT * FROM incidents 
    {where_clause}
    ORDER BY timestamp DESC
    LIMIT ? OFFSET ?
    """
    params.extend([limit, offset])

    results: List[Dict[str, Any]] = []
    with get_connection(db_path) as conn:
        cursor = conn.execute(query, params)
        for row in cursor.fetchall():
            res = dict(row)
            try:
                res["detected_classes"] = json.loads(res["detected_classes"])
            except Exception:
                res["detected_classes"] = []
            try:
                res["raw_detections"] = json.loads(res["raw_detections"])
            except Exception:
                res["raw_detections"] = []
            if res.get("human_verified_classes"):
                try:
                    res["human_verified_classes"] = json.loads(res["human_verified_classes"])
                except Exception:
                    pass
            results.append(res)
    return results


def update_review_status(
    inc_uuid: str,
    review_status: str,
    human_verified_classes: Optional[List[str]] = None,
    human_urgency_modifier: float = 1.0,
    reviewer_notes: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> bool:
    """Update human verification status and optional modifier for an incident."""
    now_str = datetime.utcnow().isoformat()
    query = """
    UPDATE incidents SET
        review_status = ?,
        human_verified_classes = ?,
        human_urgency_modifier = ?,
        reviewer_notes = ?,
        reviewed_at = ?
    WHERE incident_uuid = ?
    """
    classes_json = json.dumps(human_verified_classes) if human_verified_classes is not None else None
    params = (review_status, classes_json, human_urgency_modifier, reviewer_notes, now_str, inc_uuid)

    with get_connection(db_path) as conn:
        cursor = conn.execute(query, params)
        conn.commit()
        return cursor.rowcount > 0


def count_nearby_incidents(
    lat: float,
    lon: float,
    radius_meters: float = config.CLUSTER_RADIUS_METERS,
    exclude_uuid: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> int:
    """
    Count verified repeat incidents within a geographic radius using Haversine formula.
    """
    query = "SELECT incident_uuid, latitude, longitude FROM incidents WHERE latitude IS NOT NULL AND longitude IS NOT NULL"
    params: List[Any] = []
    if exclude_uuid:
        query += " AND incident_uuid != ?"
        params.append(exclude_uuid)

    nearby_count = 0
    with get_connection(db_path) as conn:
        cursor = conn.execute(query, params)
        for row in cursor.fetchall():
            r_lat = row["latitude"]
            r_lon = row["longitude"]
            # Haversine distance
            dist = haversine_distance_meters(lat, lon, r_lat, r_lon)
            if dist <= radius_meters:
                nearby_count += 1
    return nearby_count


def haversine_distance_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two GPS points in meters."""
    R = 6371000.0  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


def get_summary_statistics(db_path: Optional[Path] = None) -> Dict[str, Any]:
    """Retrieve high-level incident summary metrics for the Overview dashboard."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM incidents")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT priority_level, count(*) FROM incidents GROUP BY priority_level")
        by_priority = {row[0]: row[1] for row in cursor.fetchall()}

        cursor.execute("SELECT review_status, count(*) FROM incidents GROUP BY review_status")
        by_review = {row[0]: row[1] for row in cursor.fetchall()}

        # Per-class detection frequencies
        cursor.execute("SELECT detected_classes FROM incidents")
        class_counts = {"D00": 0, "D10": 0, "D20": 0, "D40": 0}
        for (dc_json,) in cursor.fetchall():
            try:
                classes = json.loads(dc_json)
                for c in classes:
                    if c in class_counts:
                        class_counts[c] += 1
            except Exception:
                pass

        return {
            "total_incidents": total,
            "by_priority": by_priority,
            "by_review": by_review,
            "by_class": class_counts,
        }


def get_incident_count(db_path: Optional[Path] = None) -> int:
    """Return total count of incidents in the database."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM incidents")
        return int(cursor.fetchone()[0])


def clear_all_incidents(confirm: bool = False, db_path: Optional[Path] = None) -> bool:
    """Safely reset incidents table when explicitly confirmed."""
    if not confirm:
        logger.warning("Attempted to clear incidents without confirmation.")
        return False
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM incidents")
        conn.commit()
    logger.info("Cleared all records from incidents table.")
    return True
