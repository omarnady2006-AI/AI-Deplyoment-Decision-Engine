"""
Copyright (c) 2026 Omar Nady

Deployment Decision Engine
Author: Omar Nady

This source code is part of a portfolio demonstration project.
Unauthorized commercial use, redistribution, or derivative work is prohibited.
See LICENSE in the project root for full terms.
"""

import yaml
from pathlib import Path

POLICY_PATH = Path("config/deployment_policy.yaml")

def load_policy():
    if not POLICY_PATH.exists():
        return {}

    try:
        with POLICY_PATH.open() as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}
