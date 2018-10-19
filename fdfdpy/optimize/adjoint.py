import numpy as np
import scipy.sparse as sp
from fdfdpy.linalg import solver_direct, grid_average
from fdfdpy.derivatives import unpack_derivs
from fdfdpy.constants import *

def adjoint_linear(simulation, b_aj, averaging=False, solver=DEFAULT_SOLVER, matrix_format=DEFAULT_MATRIX_FORMAT):
    # Compute the adjoint field for a linear problem
    # Note: the correct definition requires simulating with the transpose matrix A.T
    EPSILON_0_ = EPSILON_0*simulation.L0
    MU_0_ = MU_0*simulation.L0
    omega = simulation.omega

    (Nx, Ny) = (simulation.Nx, simulation.Ny)
    M = Nx*Ny
    A = simulation.A

    ez = solver_direct(A.T, b_aj, solver=solver)
    Ez = ez.reshape((Nx, Ny))

    return Ez