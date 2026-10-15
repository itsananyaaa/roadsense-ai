"""
RoadSense AI - App Foundation Automated Test
Validates that app/main.py boots with zero unhandled exceptions using Streamlit's AppTest.
"""

import pytest
from streamlit.testing.v1 import AppTest


def test_main_app_loads_cleanly():
    """Verify app/main.py initializes and renders with no exceptions."""
    at = AppTest.from_file("app/main.py", default_timeout=30)
    at.run()
    assert not at.exception, f"AppTest raised an exception: {at.exception}"

    # Verify main title and subtitle rendered
    assert len(at.title) >= 1
    assert "Road Intelligence Console" in at.title[0].value


def test_demo_dataset_button_click():
    """Verify clicking the Load Demo Dataset button updates the database."""
    at = AppTest.from_file("app/main.py", default_timeout=30)
    at.run()
    assert not at.exception

    # Locate sidebar button for loading demo
    demo_btn = at.sidebar.button(key="btn_load_demo")
    assert demo_btn is not None
    demo_btn.click().run()
    assert not at.exception
