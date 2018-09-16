
# This adds the pyMKL directory to the path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'pyMKL')))

# This line makes it possible to load Simulation object directly as
# `from fdfdpy import Simulation`
from .simulation import Simulation
