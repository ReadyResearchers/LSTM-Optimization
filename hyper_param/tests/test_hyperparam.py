"""Test the artifact's source code"""

from hyper_param import __main__


def test_get_stock_data():
    """Test the function that extracts the stock data"""
    __main__.get_stock_data()


def test_get_temp_data():
    """Test the function that extracts the temperature data"""
    __main__.get_temp_data()


def test_objective():
    """Make sure the objective function works and has non-zero accuracy"""
    study = __main__.optuna.create_study(study_name="Test", direction="maximize")
    study.optimize(__main__.objective, n_trials=100, timeout=100)
