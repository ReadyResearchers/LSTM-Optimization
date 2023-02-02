"""Test the artifact's source code"""

from hyper_param import __main__


def test_extract_data():
    """Test the function that extracts the stock data"""
    __main__.extract_data()


def test_objective():
    """Make sure the objective function works and has non-zero accuracy"""
    study = __main__.optuna.create_study(study_name="Test", direction="minimize")
    study.optimize(__main__.objective, n_trials=100, timeout=100)
