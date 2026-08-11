from __future__ import annotations

import platform
import sys
from importlib.metadata import version

PACKAGES = [
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "xgboost",
    "shap",
    "matplotlib",
    "PyYAML",
    "joblib",
    "ucimlrepo",
]


def main() -> None:
    print("=" * 70)
    print("CLINICAL READMISSION PROJECT - ENVIRONMENT CHECK")
    print("=" * 70)

    print(f"Python version : {sys.version.split()[0]}")
    print(f"Python path    : {sys.executable}")
    print(f"Platform       : {platform.platform()}")

    print("\nPackage versions")
    print("-" * 70)

    for package in PACKAGES:
        print(f"{package:<20} {version(package)}")

    print("\nEnvironment check completed successfully.")


if __name__ == "__main__":
    main()
