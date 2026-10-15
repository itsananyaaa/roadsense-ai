"""
RoadSense AI - Geospatial Intelligence & EXIF Metadata Extractor
Handles EXIF GPS extraction with strict provenance labeling.
Enforces Non-Negotiable Integrity Rule: Never invent or hallucinate GPS coordinates.
"""

from enum import Enum
from typing import Dict, Any, Optional, Tuple
from PIL import Image, ExifTags

from core.logger import get_logger

logger = get_logger("roadsense.geo")


class LocationProvenance(str, Enum):
    """Explicit provenance taxonomy for all geospatial coordinates."""
    EXIF_GPS = "EXIF_GPS"
    USER_ENTERED = "USER_ENTERED"
    EXPLICIT_SYNTHETIC_DEMO = "EXPLICIT_SYNTHETIC_DEMO"
    UNKNOWN = "UNKNOWN"


def extract_exif_gps(image_path: str) -> Tuple[Optional[float], Optional[float], str]:
    """
    Extract latitude and longitude from image EXIF metadata if present.
    Returns: (latitude, longitude, provenance)
    If no EXIF GPS exists, returns (None, None, 'UNKNOWN').
    """
    try:
        with Image.open(image_path) as img:
            exif_raw = img._getexif()
            if not exif_raw:
                return None, None, "UNKNOWN"

            # Resolve GPSInfo tag
            gps_info = None
            for tag_id, value in exif_raw.items():
                tag_name = ExifTags.TAGS.get(tag_id, tag_id)
                if tag_name == "GPSInfo":
                    gps_info = value
                    break

            if not gps_info:
                return None, None, "UNKNOWN"

            # Parse GPS tags
            gps_tags = {}
            for t_id, t_val in gps_info.items():
                t_name = ExifTags.GPSTAGS.get(t_id, t_id)
                gps_tags[t_name] = t_val

            # Require Latitude and Longitude
            lat_data = gps_tags.get("GPSLatitude")
            lat_ref = gps_tags.get("GPSLatitudeRef")
            lon_data = gps_tags.get("GPSLongitude")
            lon_ref = gps_tags.get("GPSLongitudeRef")

            if not (lat_data and lat_ref and lon_data and lon_ref):
                return None, None, "UNKNOWN"

            lat = _convert_to_degrees(lat_data)
            if lat_ref.upper() != "N":
                lat = -lat

            lon = _convert_to_degrees(lon_data)
            if lon_ref.upper() != "E":
                lon = -lon

            # Boundary checks
            if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                logger.info(f"Extracted valid EXIF GPS: ({lat:.6f}, {lon:.6f}) from {image_path}")
                return round(lat, 6), round(lon, 6), "EXIF_GPS"

    except Exception as e:
        logger.debug(f"Could not extract EXIF GPS from {image_path}: {e}")

    return None, None, "UNKNOWN"


def _convert_to_degrees(dms_tuple) -> float:
    """Helper to convert EXIF degrees, minutes, seconds tuple to decimal degrees."""
    def _val(x):
        try:
            return float(x)
        except Exception:
            return float(x[0]) / float(x[1])

    deg = _val(dms_tuple[0])
    minute = _val(dms_tuple[1])
    sec = _val(dms_tuple[2])
    return deg + (minute / 60.0) + (sec / 3600.0)


def format_coordinates(lat: Optional[float], lon: Optional[float], provenance: str = "UNKNOWN") -> str:
    """Format coordinates for human display with explicit provenance label."""
    if lat is None or lon is None:
        return "Location Not Available (Unknown)"
    lat_dir = "N" if lat >= 0 else "S"
    lon_dir = "E" if lon >= 0 else "W"
    prov_badge = f"[{provenance.replace('_', ' ')}]"
    return f"{abs(lat):.5f}° {lat_dir}, {abs(lon):.5f}° {lon_dir} {prov_badge}"
