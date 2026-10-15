"""
RoadSense AI - Web Application Runner
Launches the high-performance modern web console powered by FastAPI and Uvicorn.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn

def main():
    print("=" * 65)
    print("  RoadSense AI — Next-Gen Municipal Road Intelligence Console")
    print("  Serving at: http://localhost:8000")
    print("=" * 65)
    uvicorn.run("web.backend:app", host="0.0.0.0", port=8000, log_level="info")

if __name__ == "__main__":
    main()
