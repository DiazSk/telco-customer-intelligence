#!/usr/bin/env python3
"""
Code formatting script for CI/CD compatibility.

Run this before committing to ensure CI passes.
"""

import os
import subprocess
import sys


def run_command(cmd, description):
    """Run a command and report results."""
    print(f"\n🔧 {description}...")
    try:
        result = subprocess.run(
            cmd, shell=True, check=True, capture_output=True, text=True
        )
        print(f"✅ {description} completed")
        if result.stdout:
            print(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        if e.stdout:
            print(f"Output: {e.stdout.strip()}")
        if e.stderr:
            print(f"Error: {e.stderr.strip()}")
        return False


def main():
    """Format code and run quality checks."""
    print("🚀 Code Formatting & Quality Check")
    print("=" * 50)

    # Change to project root
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    commands = [
        ("python -m black .", "Black formatting"),
        ("python -m isort .", "Import sorting"),
        ("python -m black --check .", "Black format verification"),
        ("python -m isort --check-only .", "Import sort verification"),
        (
            "python -m flake8 src/ scripts/ tests/ --max-line-length=120 --extend-ignore=E203,W503,E402",
            "Flake8 linting",
        ),
    ]

    failed = []
    for cmd, desc in commands:
        if not run_command(cmd, desc):
            failed.append(desc)

    print("\n" + "=" * 50)
    if not failed:
        print("🎉 All checks passed! Ready to commit.")
        return 0
    else:
        print(f"❌ {len(failed)} check(s) failed:")
        for fail in failed:
            print(f"  - {fail}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
