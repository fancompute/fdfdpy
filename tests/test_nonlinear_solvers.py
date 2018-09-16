import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fdfdpy import Simulation
from numpy import pi, ones, zeros, square, conj, logspace, array, append, unwrap, angle
from numpy.testing import assert_allclose

n0 = 3.4
omega = 2*pi*200e12
dl = 0.01
chi3 = 2.8E-18

width  = 1
L      = 5
L_chi3 = 4

width_voxels  = int(width/dl)
L_chi3_voxels = int(L_chi3/dl)

Nx = int(L/dl)
Ny = int(3.5*width/dl)

eps_r = ones((Nx, Ny))
eps_r[:,int(Ny/2-width_voxels/2):int(Ny/2+width_voxels/2)] = square(n0)

nl_region = zeros(eps_r.shape)
nl_region[int(Nx/2-L_chi3_voxels/2):int(Nx/2+L_chi3_voxels/2), int(Ny/2-width_voxels/2):int(Ny/2+width_voxels/2)] = 1

simulation = Simulation(omega, eps_r, dl, [15, 15], 'Ez')
simulation.add_mode(n0, 'x', [17, int(Ny/2)], width_voxels*3)
simulation.setup_modes()
kerr_nonlinearity = lambda e: 3*chi3/square(simulation.L0)*square(abs(e))
dkerr_de = lambda e: 3*chi3/square(simulation.L0)*conj(e)

srcval_vec = logspace(1, 3, 3)
pwr_vec = array([])
T_vec = array([])
for srcval in srcval_vec:
    simulation.setup_modes()
    simulation.src *= srcval
    
    # Newton
    simulation.solve_fields_nl(kerr_nonlinearity, nl_region,
                               dnl_de=dkerr_de, timing=False, averaging=True,
                               Estart=None, solver_nl='newton')
    E_newton = simulation.fields["Ez"]
    
    # Born
    simulation.solve_fields_nl(kerr_nonlinearity, nl_region,
                               dnl_de=dkerr_de, timing=False, averaging=True,
                               Estart=None, solver_nl='born')
    E_born = simulation.fields["Ez"]

    # More solvers (if any) should be added here with corresponding calls to assert_allclose() below
    
    assert_allclose(E_newton, E_born, rtol=1e-3)
