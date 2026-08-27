#!/usr/bin/env python3
# ruff: noqa: T201
"""
Test runner script for ExoSim2.0.

This script provides convenient ways to run different categories of tests
and generate coverage reports using the modern test suite structure.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def check_dependency(module_name: str, package_name: str | None = None) -> bool:
    """Check if a Python package is available."""
    try:
        subprocess.run(
            [sys.executable, "-c", f"import {module_name}"],
            capture_output=True,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, ImportError):
        return False


def run_pytest(test_paths: list[str], **options) -> int:
    """Run pytest with specified options."""
    cmd = [sys.executable, "-m", "pytest"]

    # Override config to avoid conflicts
    cmd.extend(["--override-ini", "addopts="])

    # Add test paths
    cmd.extend(test_paths)

    # Add verbosity
    if options.get("verbose"):
        cmd.append("-v")

    # Add markers for filtering
    if options.get("markers"):
        cmd.extend(["-m", options["markers"]])

    # Add parallel execution
    if options.get("parallel") and check_dependency("xdist"):
        cmd.extend(["-n", "auto"])
    elif options.get("parallel"):
        print("Warning: pytest-xdist not available, running sequentially")

    # Add coverage
    if options.get("coverage") and check_dependency("pytest_cov"):
        cmd.extend(
            [
                "--cov=src/exosim",
                "--cov-report=html:htmlcov",
                "--cov-report=term-missing",
                "--cov-report=xml",
            ]
        )
    elif options.get("coverage"):
        print("Warning: pytest-cov not available, skipping coverage")

    # Add specific test pattern
    if options.get("pattern"):
        cmd.extend(["-k", options["pattern"]])

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def main() -> None:
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(
        description="Run ExoSim2.0 tests",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--category",
        "-c",
        choices=["unit", "integration", "all", "legacy"],
        default="unit",
        help="Category of tests to run",
    )

    parser.add_argument(
        "--module",
        "-m",
        choices=[
            "core",
            "models",
            "tasks",
            "utils",
            "tools",
            "output",
            "plots",
            "recipes",
        ],
        help="Run tests for a specific module",
    )

    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report",
    )

    parser.add_argument(
        "--fast",
        "-f",
        action="store_true",
        help="Run only fast tests (exclude slow tests)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    parser.add_argument(
        "--parallel",
        "-p",
        action="store_true",
        help="Run tests in parallel (requires pytest-xdist)",
    )

    parser.add_argument(
        "--pattern",
        "-k",
        help="Run tests matching this pattern",
    )

    parser.add_argument(
        "path",
        nargs="?",
        help="Specific test file or directory to run",
    )

    args = parser.parse_args()

    # Check if pytest is available
    if not check_dependency("pytest"):
        print("ERROR: pytest not available. Please install pytest:")
        print(f"{sys.executable} -m pip install pytest pytest-cov pytest-xdist")
        sys.exit(1)

    # Determine test paths
    base_dir = Path(__file__).parent
    test_paths = []

    if args.path:
        # Specific path provided
        test_paths = [args.path]
    elif args.category == "legacy":
        # Run legacy tests
        if (base_dir / "tests").exists():
            test_paths = ["tests/"]
        else:
            print("No legacy tests directory found")
            sys.exit(1)
    elif args.category == "all":
        # Run all new tests
        test_paths = ["test_suite/"]
    elif args.module:
        # Run specific module tests
        module_path = base_dir / "test_suite" / "unit" / args.module
        if module_path.exists():
            test_paths = [str(module_path)]
        else:
            print(f"Module test directory not found: {module_path}")
            sys.exit(1)
    else:
        # Default to unit tests
        if args.category == "unit":
            test_paths = ["test_suite/unit/"]
        elif args.category == "integration":
            test_paths = ["test_suite/integration/"]

    if not test_paths:
        print("No test paths determined")
        sys.exit(1)

    # Build options
    options = {
        "verbose": args.verbose,
        "parallel": args.parallel,
        "coverage": args.coverage,
        "pattern": args.pattern,
    }

    # Add fast filter
    if args.fast:
        options["markers"] = "not slow"

    # Run tests
    print(f"Running {args.category} tests...")
    if args.coverage:
        print("Coverage report will be generated in htmlcov/")

    exit_code = run_pytest(test_paths, **options)

    if exit_code == 0:
        print("\n✅ Tests completed successfully!")
        if args.coverage and Path("htmlcov/index.html").exists():
            print("📊 Coverage report available at htmlcov/index.html")
    else:
        print("\n❌ Tests failed!")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
