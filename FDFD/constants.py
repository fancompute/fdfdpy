import numpy as np

EPSILON_0 = 8.854e-12
MU_0 = np.pi*4e-7
C_0 = np.sqrt(1/EPSILON_0/MU_0)
ETA_0 = np.sqrt(MU_0/EPSILON_0)

DEFAULT_MATRIX_FORMAT = 'csr'
DEFAULT_SOLVER = 'pardiso'
DEFAULT_LENGTH_SCALE = 1e-6 # microns