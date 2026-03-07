"""
scripts/start_server.py
-----------------------
Start the Deployment Decision Engine API server.

Usage:
    py -3 scripts/start_server.py
    py -3 scripts/start_server.py --port 9000
    py -3 scripts/start_server.py --host 0.0.0.0 --port 8000
"""
from __future__ import annotations
import argparse
import sys
import webbrowser
from pathlib import Path
from threading import Timer

# Ensure project root is in sys.path so `src` package resolves
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the Deployment Decision Engine server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Bind port (default: 8080)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        sys.exit("uvicorn is not installed. Run: pip install uvicorn")

    from src.api.gui_app import app  # noqa: PLC0415

    url = f"http://{args.host}:{args.port}"
    print("=" * 70)
    print("Deployment Decision Engine — API Server")
    print("=" * 70)
    print(f"\n  Listening on:  {url}")
    print("  Press Ctrl+C to stop\n")

    if not args.no_browser:
        Timer(1.5, lambda: webbrowser.open(url)).start()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
