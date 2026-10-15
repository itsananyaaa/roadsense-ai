"""
RoadSense AI - Automated UI Tests Across All Console Pages
Ensures every application page initializes cleanly with AppTest without unhandled exceptions.
"""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

from core.db import init_db
from core.demo import seed_demo_database


@pytest.fixture(scope="module", autouse=True)
def ensure_db_ready():
    """Ensure database has schema and demo records for pages to render."""
    init_db()
    seed_demo_database(reset_existing=False)


def test_page_1_detect_damage():
    """Verify Page 1: Detect Damage launches cleanly."""
    at = AppTest.from_file("app/pages/1_Detect_Damage.py", default_timeout=15)
    at.run()
    assert not at.exception, f"Page 1 raised exception: {at.exception}"


def test_page_2_video_intelligence():
    """Verify Page 2: Video Intelligence launches cleanly."""
    at = AppTest.from_file("app/pages/2_Video_Intelligence.py", default_timeout=15)
    at.run()
    assert not at.exception, f"Page 2 raised exception: {at.exception}"


def test_page_3_incident_map():
    """Verify Page 3: Incident Map launches cleanly."""
    at = AppTest.from_file("app/pages/3_Incident_Map.py", default_timeout=15)
    at.run()
    assert not at.exception, f"Page 3 raised exception: {at.exception}"


def test_page_4_priority_center():
    """Verify Page 4: Priority Center launches cleanly."""
    at = AppTest.from_file("app/pages/4_Priority_Center.py", default_timeout=15)
    at.run()
    assert not at.exception, f"Page 4 raised exception: {at.exception}"


def test_page_5_model_lab():
    """Verify Page 5: Model Lab launches cleanly and reads reports/ artifacts."""
    at = AppTest.from_file("app/pages/5_Model_Lab.py", default_timeout=15)
    at.run()
    assert not at.exception, f"Page 5 raised exception: {at.exception}"


def test_page_6_reports_history():
    """Verify Page 6: Reports & History launches cleanly."""
    at = AppTest.from_file("app/pages/6_Reports_History.py", default_timeout=15)
    at.run()
    assert not at.exception, f"Page 6 raised exception: {at.exception}"


def test_page_7_about_methodology():
    """Verify Page 7: About & Methodology launches cleanly."""
    at = AppTest.from_file("app/pages/7_About_Methodology.py", default_timeout=15)
    at.run()
    assert not at.exception, f"Page 7 raised exception: {at.exception}"
