"""
Check if all required libraries for model training and tuning are installed.
Run this script before running any model scripts.
"""

import importlib
import sys

REQUIRED = [
    'torch',
    'transformers',
    'datasets',
    'pandas',
    'optuna',
    'sklearn',
    'matplotlib',
    'seaborn',
    'notebook',
]

missing = []
for pkg in REQUIRED:
    try:
        importlib.import_module(pkg)
    except ImportError:
        missing.append(pkg)

if missing:
    print("Missing packages detected:")
    for pkg in missing:
        print(f"  - {pkg}")
    print("\nPlease install them with:")
    print(f"pip install {' '.join(missing)}")
    sys.exit(1)
else:
    print("All required packages are installed.")
