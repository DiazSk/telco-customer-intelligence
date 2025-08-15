#!/usr/bin/env python3
"""
Script to check if all CI requirements are met.

This helps debug CI issues locally before pushing.
"""

import os
import sys
from pathlib import Path


def check_file_exists(filepath, required=True):
    """Check if a file exists and report status."""
    if os.path.exists(filepath):
        print(f"✓ {filepath} exists")
        return True
    else:
        status = "✗" if required else "⚠"
        print(f"{status} {filepath} missing {'(required)' if required else '(optional)'}")
        return False


def check_directory_exists(dirpath, required=True):
    """Check if a directory exists and report status."""
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        print(f"✓ {dirpath}/ exists")
        return True
    else:
        status = "✗" if required else "⚠"
        print(f"{status} {dirpath}/ missing {'(required)' if required else '(optional)'}")
        return False


def main():
    """Run all CI requirement checks."""
    print("=" * 60)
    print("CI/CD REQUIREMENTS CHECK")
    print("=" * 60)

    issues = []

    # Check required files
    required_files = [
        "requirements.txt",
        "requirements-dev.txt",
        "configs/pipeline_config.yaml",
        ".github/workflows/ci.yml",
        ".pre-commit-config.yaml",
    ]

    print("\n1. Checking required files:")
    for file in required_files:
        if not check_file_exists(file):
            issues.append(f"Missing required file: {file}")

    # Check required directories
    required_dirs = ["src", "tests", "tests/unit", "scripts", "configs"]

    print("\n2. Checking required directories:")
    for dir in required_dirs:
        if not check_directory_exists(dir):
            issues.append(f"Missing required directory: {dir}")

    # Check optional directories that CI might need
    optional_dirs = [
        "data",
        "data/raw",
        "data/processed",
        "data/features",
        "logs",
        "tests/integration",
    ]

    print("\n3. Checking optional directories:")
    for dir in optional_dirs:
        check_directory_exists(dir, required=False)

    # Check if key Python files exist
    key_files = [
        "src/__init__.py",
        "src/api/__init__.py",
        "src/data_pipeline/__init__.py",
        "tests/__init__.py",
        "tests/unit/test_pipeline.py",
    ]

    print("\n4. Checking key Python files:")
    for file in key_files:
        if not check_file_exists(file, required=False):
            issues.append(f"Missing key file: {file}")

    # Check Python imports
    print("\n5. Checking Python imports:")
    try:
        import pytest

        print("✓ pytest available")
    except ImportError:
        print("✗ pytest not available")
        issues.append("pytest not installed")

    try:
        import pandas

        print("✓ pandas available")
    except ImportError:
        print("✗ pandas not available")
        issues.append("pandas not installed")

    try:
        import fastapi

        print("✓ FastAPI available")
    except ImportError:
        print("✗ FastAPI not available")
        issues.append("FastAPI not installed")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if not issues:
        print("✓ All CI requirements met!")
        return 0
    else:
        print(f"✗ Found {len(issues)} issues:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nFix these issues before running CI/CD pipeline.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
