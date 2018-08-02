import scipy.sparse as sp
import scipy.sparse.linalg as spl
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'pyMKL')))
from pyMKL import pardisoSolver

import numpy as np
from time import time

from FDFD.constants import *
from FDFD.pml import S_create

def dL(N, xrange, yrange=None):
	# solves for the grid spacing
	
	if yrange is None:
		L = np.array([np.diff(xrange)[0]])  # Simulation domain lengths
	else:
		L = np.array([np.diff(xrange)[0],
				   np.diff(yrange)[0]])  # Simulation domain lengths
	return L/N


def is_equal(matrix1, matrix2):
	# checks if two sparse matrices are equal

	return (matrix1!=matrix2).nnz==0


def construct_A(omega, xrange, yrange, eps_r, NPML, pol,
				averaging=False,
				timing=False,
				matrix_format=DEFAULT_MATRIX_FORMAT):
	# makes the A matrix
	N = np.asarray(eps_r.shape)  # Number of mesh cells
	M = np.prod(N)  # Number of unknowns
	
	if pol == 'Ez':
		vector_eps_z = EPSILON_0*eps_r.reshape((-1,))
		T_eps_z = sp.spdiags(vector_eps_z, 0, M, M, format=matrix_format)

		(Sxf, Sxb, Syf, Syb) = S_create(omega, N, NPML, xrange, yrange, matrix_format=matrix_format)
		
		# Construct derivate matrices
		Dyb = Syb.dot(createDws('y', 'b', dL(N, xrange, yrange), N, matrix_format=matrix_format))
		Dxb = Sxb.dot(createDws('x', 'b', dL(N, xrange, yrange), N, matrix_format=matrix_format))
		Dxf = Sxf.dot(createDws('x', 'f', dL(N, xrange, yrange), N, matrix_format=matrix_format))
		Dyf = Syf.dot(createDws('y', 'f', dL(N, xrange, yrange), N, matrix_format=matrix_format))
		
		A = (Dxf*1/MU_0).dot(Dxb) \
			+ (Dyf*1/MU_0).dot(Dyb) \
			+ omega**2*T_eps_z
		A = A / (omega**2*EPSILON_0)        # normalize A to be unitless.  (note, this isn't in original fdfdpy)

		# Construct derivative of A with respect to epsilon
		# Note: the derivative with respect to each epsilon_i is a matrix with a single non-zero element at position ii
		# These can therefore be just lumped into one diagonal matrix 

		dAdeps = omega**2*EPSILON_0*sp.eye(M, M, 0, format=matrix_format)
		dAdeps = dAdeps / (omega**2*EPSILON_0)        # normalize A to be unitless.  (note, this isn't in original fdfdpy)

			
	elif pol == 'Hz':
		# Note, haven't included grid_average function yet
		if averaging:
			vector_eps_x = grid_average(EPSILON_0*eps_r, 'x').reshape((-1,))
			vector_eps_y = grid_average(EPSILON_0*eps_r, 'y').reshape((-1,))
		else:
			vector_eps_x = EPSILON_0*eps_r.reshape((-1,))
			vector_eps_y = EPSILON_0*eps_r.reshape((-1,))

		# Setup the T_eps_x, T_eps_y, T_eps_x_inv, and T_eps_y_inv matrices
		T_eps_x = sp.spdiags(vector_eps_x, 0, M, M, format=matrix_format)
		T_eps_y = sp.spdiags(vector_eps_y, 0, M, M, format=matrix_format)
		T_eps_x_inv = sp.spdiags(1/vector_eps_x, 0, M, M, format=matrix_format)
		T_eps_y_inv = sp.spdiags(1/vector_eps_y, 0, M, M, format=matrix_format)
		
		(Sxf, Sxb, Syf, Syb) = S_create(omega, N, NPML, xrange, yrange, matrix_format=matrix_format)
		
		# Construct derivate matrices
		Dyb = Syb.dot(createDws('y', 'b', dL(N, xrange, yrange), N, matrix_format=matrix_format))
		Dxb = Sxb.dot(createDws('x', 'b', dL(N, xrange, yrange), N, matrix_format=matrix_format))
		Dxf = Sxf.dot(createDws('x', 'f', dL(N, xrange, yrange), N, matrix_format=matrix_format))
		Dyf = Syf.dot(createDws('y', 'f', dL(N, xrange, yrange), N, matrix_format=matrix_format))
		
		A = Dxf.dot(T_eps_x_inv).dot(Dxb) \
			+ Dyf.dot(T_eps_y_inv).dot(Dyb) \
			+ omega**2*MU_0*sp.eye(M)
		
		A = A / (omega**2*MU_0)     # normalize A to be unitless.  (note, this isn't in original fdfdpy)

		# Construct derivative of A with respect to epsilon
		# Note: the derivative with respect to each epsilon_i is a matrix with a single non-zero element at position ii
		# These can therefore be just lumped into one diagonal matrix 
		# Also note: this is not straightforwardly clear for this polarization, but it seems to be correct. 
		# The requirement is that Dxf and Dxb.T have no overlapping non-zero elements apart from on the diagonal (same for Dyf and Dyb)

		dAdepsx = -EPSILON_0*Dxf.dot(np.square(T_eps_x_inv)).dot(Dxb)
		dAdepsx = dAdepsx / (omega**2*MU_0)        # normalize A to be unitless.  (note, this isn't in original fdfdpy)
		dAdepsy= -EPSILON_0*Dyf.dot(np.square(T_eps_y_inv)).dot(Dyb)
		dAdepsy = dAdepsy / (omega**2*MU_0)        # normalize A to be unitless.  (note, this isn't in original fdfdpy)

		dAdeps = {
			'dAdepsx' : dAdepsx,
			'dAdepsy' : dAdepsy
		}

		
	else:
		raise ValueError("something went wrong and pol is not one of Ez, Hz, instead was given {}".format(pol))
		
	derivs = {
		'Dyb' : Dyb,
		'Dxb' : Dxb,
		'Dxf' : Dxf,
		'Dyf' : Dyf
	}
					
	return (A, derivs, dAdeps)


def createDws(w, s, dL, N, matrix_format=DEFAULT_MATRIX_FORMAT):
	# creates the derivative matrices

	Nx = N[0]
	dx = dL[0]
	if len(N) is not 1:
		Ny = N[1]
		dy = dL[1]
	else:
		Ny = 1
		dy = inf
	if w is 'x':
		if s is 'f':
			dxf = sp.diags([-1, 1, 1], [0, 1, -Nx+1], shape=(Nx, Nx))
			Dws = 1/dx*sp.kron(sp.eye(Ny), dxf, format=matrix_format)
		else:
			dxb = sp.diags([1, -1, -1], [0, -1, Nx-1], shape=(Nx, Nx))
			Dws = 1/dx*sp.kron(sp.eye(Ny), dxb, format=matrix_format)
	if w is 'y':
		if s is 'f':
			dyf = sp.diags([-1, 1, 1], [0, 1, -Ny+1], shape=(Ny, Ny))
			Dws = 1/dy*sp.kron(dyf, sp.eye(Nx), format=matrix_format)
		else:
			dyb = sp.diags([1, -1, -1], [0, -1, Ny-1], shape=(Ny, Ny))
			Dws = 1/dy*sp.kron(dyb, sp.eye(Nx), format=matrix_format)
	return Dws


def unpack_derivs(derivs):
	# takes derivs dictionary and returns tuple for convenience

	Dyb = derivs['Dyb']
	Dxb = derivs['Dxb']
	Dxf = derivs['Dxf']
	Dyf = derivs['Dyf']
	return (Dyb, Dxb, Dxf, Dyf)


def solver_eigs(A, Neigs, guess_value=0, guess_vector=None, timing=False):
	# solves for the eigenmodes of A

	if timing:
		start = time()
	(values, vectors) = spl.eigs(A, k=Neigs, sigma=guess_value, v0=guess_vector, which='LM')
	if timing:
		end = time()
		print('Elapsed time for eigs() is %.4f secs' % (end - start))
	return (values, vectors)


def solver_direct(A, b, timing=False, solver=DEFAULT_SOLVER):
	# solves linear system of equations
	
	b = b.astype(np.complex128)
	b = b.reshape((-1,))

	if not b.any():
		return zeros(b.shape)

	if timing:
		t = time()

	if solver.lower() == 'pardiso':
		pSolve = pardisoSolver(A, mtype=13) # Matrix is complex unsymmetric due to SC-PML
		pSolve.factor()
		x = pSolve.solve(b)
		pSolve.clear()

	elif solver.lower() == 'scipy':
		x = spl.spsolve(A, b)

	else:
		raise ValueError('Invalid solver choice: {}, options are pardiso or scipy'.format(str(solver)))

	if timing:
		print('Linear system solve took {:.2f} seconds'.format(time()-t))

	return x

