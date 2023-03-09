"""Test the artifact's source code"""

import runpy
from experiment import __main__


def test_experiment():
    """Make sure the whole script works as intended"""
    runpy.run_path("experiment/__main__.py", run_name="__main__")
