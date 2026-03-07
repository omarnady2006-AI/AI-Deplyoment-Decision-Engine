"""
Copyright (c) 2026 Omar Nady

Deployment Decision Engine
Author: Omar Nady

This source code is part of a portfolio demonstration project.
Unauthorized commercial use, redistribution, or derivative work is prohibited.
See LICENSE in the project root for full terms.
"""

import json
import time
from pathlib import Path

LOG_FILE = Path("analysis_log.jsonl")

def log_analysis_event(data: dict):
    entry = {
        "timestamp": time.time(),
        **data
    }

    with LOG_FILE.open("a") as f:
        f.write(json.dumps(entry) + "\n")
