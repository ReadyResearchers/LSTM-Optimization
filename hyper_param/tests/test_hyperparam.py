"""Test the artifact's source code"""

from hyper_param import __main__
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_objective():
    """Make sure the objective function works and has non-zero accuracy"""
    study = __main__.optuna.create_study(study_name="Test", direction="minimize")
    study.optimize(__main__.objective, n_trials=100, timeout=30)

def test_experiment():
    """Make sure the whole script works as intended"""
    os.system("python experiment/__main__.py")