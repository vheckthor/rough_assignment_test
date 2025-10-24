import os
import sys

# Allow running as a script: python bayesian_net/run_tests.py
if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bayesian_net.experiment import run_experiment


def main():
    # default test on heart.csv in project root or in bayesian_net/data/raw
    candidates = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'raw', 'heart.csv')),
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'heart.csv')),
    ]
    csv_path = next((p for p in candidates if os.path.isfile(p)), None)
    if csv_path is None:
        raise FileNotFoundError("Could not find heart.csv in bayesian_net/data/raw or project root.")

    print(f"Running BN experiment on {csv_path} ...")
    run_experiment(csv_path=csv_path, target_col=None, n_splits=5, seed=42, bins=5)


if __name__ == '__main__':
    main()
