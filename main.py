#!/usr/bin/env python3
"""
Deployment Decision Engine - Main Entry Point

A verifiable safety system for deployment decision making.

Usage:
    python main.py --cli <model.onnx>
    python main.py --gui
    python main.py --api
    python main.py --supervisor
    python main.py --validate
    python main.py --all
    
GUI Usage:
    python main.py --gui                    # Open GUI on default port 8080
    python main.py --gui --port 9000       # Open GUI on custom port
"""

from __future__ import annotations

import sys
import argparse
import asyncio
import webbrowser
from pathlib import Path
from threading import Timer


def run_cli(model_path: str, profile_path: str | None = None) -> int:
    """Run CLI pipeline."""
    from src.cli.main import main as cli_main
    sys.argv = ["deploycheck", model_path]
    if profile_path:
        sys.argv.extend(["--profile", profile_path])
    return cli_main()


def run_gui(port: int = 8080) -> None:
    """Run GUI server and open browser."""
    import uvicorn
    from src.api.gui_app import app
    
    # Get the project root directory
    project_root = Path(__file__).resolve().parent
    
    # Configure uvicorn to serve static files
    config = uvicorn.Config(
        app, 
        host="127.0.0.1", 
        port=port,
        log_level="info"
    )
    
    server = uvicorn.Server(config)
    
    def open_browser():
        """Open browser after server starts."""
        webbrowser.open(f"http://127.0.0.1:{port}")
    
    # Open browser after a short delay
    Timer(1.5, open_browser).start()
    
    print("=" * 70)
    print("Deployment Decision Engine - GUI Mode")
    print("=" * 70)
    print(f"\nOpening GUI at http://127.0.0.1:{port}")
    print("\nPress Ctrl+C to stop the server\n")
    
    server.run()


def run_api(port: int = 8080) -> None:
    """Run API server."""
    import uvicorn
    from src.api.gui_app import app
    uvicorn.run(app, host="0.0.0.0", port=port)


def run_supervisor() -> None:
    """Run supervisor."""
    raise RuntimeError("Supervisor mode is unavailable in this build")


def run_validation() -> int:
    """Run full validation."""
    from src.cli.main import main as cli_main
    sys.argv = ["deploycheck", "--validate"]
    return cli_main()


def run_reevaluation() -> int:
    """Run reevaluation."""
    from src.cli.main import main as cli_main
    sys.argv = ["deploycheck", "--reevaluate"]
    return cli_main()


def run_all() -> None:
    """Run all components for testing."""
    print("=" * 70)
    print("Deployment Decision Engine - Full System Test")
    print("=" * 70)
    
    # Test 1: Run CLI on sample model
    print("\n[1] Testing CLI...")
    from src.cli.main import run_pipeline
    from src.core.calibration_history import load_history
    history = load_history(".deploycheck_calibration_history.json")
    print(f"   Calibration history: {len(history)} records")
    print("   [OK] CLI module loaded successfully")
    
    # Test 2: Verify API imports
    print("\n[2] Testing API...")
    from src.api.gui_app import app as _api_app
    print("   [OK] API module loaded successfully")
    
    # Test 3: Verify core imports
    print("\n[3] Testing Core...")
    from src.core.runtime_selector import recommend_runtime
    print("   [OK] Core modules loaded successfully")
    
    # Test 4: Verify validation imports
    print("\n[4] Testing Validation...")
    from src.validation.runtime_validator import validate_runtime
    from src.validation.interpret_validation import interpret_validation_result
    print("   [OK] Validation modules loaded successfully")
    
    # Test 5: Verify runtime imports
    print("\n[5] Testing Runtime...")
    try:
        from src.runtime.guardrails import SafetyEvent
        print("   [OK] Runtime modules loaded successfully")
    except ImportError as e:
        print(f"   [WARN] Runtime modules (optional): {e}")
    
    # Test 6: Verify rules
    print("\n[6] Testing Rules...")
    from src.rules.unsupported_operator import UnsupportedOperatorRule
    print("   [OK] Rules loaded successfully")
    
    print("\n" + "=" * 70)
    print("All components loaded successfully!")
    print("=" * 70)


def run_quick_test() -> None:
    """Run quick smoke test."""
    print("=" * 70)
    print("Quick Smoke Test")
    print("=" * 70)
    
    tests = [
        ("CLI", "src.cli.main", "run_pipeline"),
        ("API", "src.api.gui_app", "app"),
        ("Core", "src.core.runtime", "RuntimeName"),
        ("Rules", "src.rules.unsupported_operator", "UnsupportedOperatorRule"),
        ("Validation", "src.validation.runtime_validator", "validate_runtime"),
    ]
    
    passed = 0
    for name, module, attr in tests:
        try:
            __import__(module)
            getattr(sys.modules[module], attr)
            print(f"   [OK] {name}")
            passed += 1
        except Exception as e:
            print(f"   [FAIL] {name}: {e}")

    try:
        __import__("src.runtime.guardrails")
        getattr(sys.modules["src.runtime.guardrails"], "SafetyEvent")
        print("   [OK] Runtime")
        passed += 1
    except ModuleNotFoundError:
        print("   [SKIP] Runtime (module removed)")
    except Exception as e:
        print(f"   [FAIL] Runtime: {e}")

    print(f"\n{passed}/{len(tests) + 1} tests passed")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Deployment Decision Engine - Verifiable Safety System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --cli model.onnx
    python main.py --gui
    python main.py --gui --port 9000
    python main.py --api
    python main.py --api --port 9000
    python main.py --supervisor
    python main.py --validate
    python main.py --all
    python main.py --test
        """
    )
    
    parser.add_argument(
        "--cli",
        metavar="MODEL",
        help="Run CLI pipeline on a model"
    )
    parser.add_argument(
        "--profile",
        metavar="FILE",
        help="Input profile JSON file"
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Run GUI server (web interface)"
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="Run API server"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="API/GUI server port (default: 8080)"
    )
    parser.add_argument(
        "--supervisor",
        action="store_true",
        help="Run supervisor process"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation pipeline"
    )
    parser.add_argument(
        "--reevaluate",
        action="store_true",
        help="Run reevaluation"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run full system test"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run quick smoke test"
    )
    
    args = parser.parse_args()
    
    # Ensure project root is in path
    project_root = Path(__file__).resolve().parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Run requested mode
    if args.cli:
        return run_cli(args.cli, args.profile)
    elif args.gui:
        run_gui(args.port)
    elif args.api:
        run_api(args.port)
    elif args.supervisor:
        run_supervisor()
    elif args.validate:
        return run_validation()
    elif args.reevaluate:
        return run_reevaluation()
    elif args.all:
        run_all()
    elif args.test:
        run_quick_test()
    else:
        # Default: show help and run quick test
        parser.print_help()
        print("\n" + "=" * 70)
        print("Running quick test...")
        print("=" * 70)
        run_quick_test()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
