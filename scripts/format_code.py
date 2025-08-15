#!/usr/bin/env python3
"""
Code formatting script for CI/CD compatibility.

Run this before committing to ensure CI passes.
"""

import os
import shlex
import subprocess  # nosec B404
import sys


def run_command(cmd, description):
    """Run a command and report results."""
    print(f"\n[RUNNING] {description}...")
    try:
        # Convert string command to list for security (avoid shell=True)
        # Use shlex.split() to properly handle quoted arguments and complex commands
        cmd_list = shlex.split(cmd) if isinstance(cmd, str) else cmd
        result = subprocess.run(cmd_list, check=True, capture_output=True, text=True)  # nosec B603
        print(f"[SUCCESS] {description} completed")
        if result.stdout:
            print(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[FAILED] {description} failed")
        if e.stdout:
            print(f"Output: {e.stdout.strip()}")
        if e.stderr:
            print(f"Error: {e.stderr.strip()}")
        return False


def main():
    """Format code and run quality checks."""
    print("Code Formatting & Quality Check")
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
        print("All checks passed! Ready to commit.")
        return 0
    else:
        print(f"[FAILED] {len(failed)} check(s) failed:")
        for fail in failed:
            print(f"  - {fail}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
