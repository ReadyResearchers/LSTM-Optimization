"""Test the artifact's source code"""

import os
import subprocess
import sys


def test_experiment():
    """Make sure the whole script works as intended"""
    subprocess.call([sys.executable, "-m", "__main__"])
    os.system("{} -m __main__".format(sys.executable))
