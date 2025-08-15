#!/usr/bin/env python3
"""
Production-ready API startup script.
Handles environment-specific configurations safely.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


def start_api():
    """Start the API with environment-specific configuration."""
    import platform

    import uvicorn

    # Environment-based configuration
    env = os.getenv("ENV", "development")
    system = platform.system()

    # Platform-optimized defaults
    if system == "Windows":
        default_host = "127.0.0.1"  # Required for Windows performance
    else:
        default_host = "0.0.0.0"  # Unix systems handle this efficiently

    # Development defaults
    config = {
        "app": "src.api.main:app",
        "host": default_host,
        "port": int(os.getenv("API_PORT", "8000")),
        "log_level": "info",
        "reload": env == "development",
        "workers": 1,
    }

    # Production overrides (external access, multiple workers)
    if env == "production":
        config.update(
            {
                "host": "0.0.0.0",  # Allow external access
                "workers": int(os.getenv("API_WORKERS", "4")),
                "log_level": "warning",
                "reload": False,
            }
        )

    # Docker/container overrides
    elif env == "docker":
        config.update(
            {
                "host": "0.0.0.0",  # Required for container networking
                "workers": 1,  # Let container orchestrator handle scaling
                "reload": False,
            }
        )

    print(f"Starting API in {env} mode on {system}...")
    print(f"   Host: {config['host']}")
    print(f"   Port: {config['port']}")
    print(f"   Workers: {config['workers']}")

    if system == "Windows" and config["host"] == "127.0.0.1":
        print("   Windows-optimized: Using localhost for performance")
    elif system in ["Darwin", "Linux"] and config["host"] == "0.0.0.0":
        print("   Unix-optimized: Using all interfaces for flexibility")

    uvicorn.run(**config)


if __name__ == "__main__":
    start_api()
