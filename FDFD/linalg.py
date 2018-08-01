import scipy.sparse as sp
import scipy.sparse.linalg as spl
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'pyMKL')))
from pyMKL import pardisoSolver

import numpy as np
from time import time

from FDFD.constants import *

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
		
	else:
		raise ValueError("something went wrong and pol is not one of Ez, Hz, instead was given {}".format(pol))
		
	derivs = {
		'Dyb' : Dyb,
		'Dxb' : Dxb,
		'Dxf' : Dxf,
		'Dyf' : Dyf
	}
					
	return (A, derivs)


def sig_w(l, dw, m=4, lnR=-12):
	# helper for S()

	sig_max = -(m+1)*lnR/(2*ETA_0*dw)
	return sig_max*(l/dw)**m


def S(l, dw, omega):
	# helper for create_sfactor()

	return 1-1j*sig_w(l, dw)/(omega*EPSILON_0)


def create_sfactor(wrange, s, omega, Nw, Nw_pml):
	# used to help construct the S matrices for the PML creation

	sfactor_array = np.ones(Nw, dtype=np.complex128)
	if Nw_pml < 1:
		return sfactor_array
	hw = np.diff(wrange)[0]/Nw
	dw = Nw_pml*hw
	for i in range(0, Nw):
		if s is 'f':
			if i <= Nw_pml:
				sfactor_array[i] = S(hw * (Nw_pml - i + 0.5), dw, omega)
			elif i > Nw - Nw_pml:
				sfactor_array[i] = S(hw * (i - (Nw - Nw_pml) - 0.5), dw, omega)
		if s is 'b':
			if i <= Nw_pml:
				sfactor_array[i] = S(hw * (Nw_pml - i + 1), dw, omega)
			elif i > Nw - Nw_pml:
				sfactor_array[i] = S(hw * (i - (Nw - Nw_pml) - 1), dw, omega)
	return sfactor_array


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


def S_create(omega, N, Npml, xrange, yrange=None, matrix_format=DEFAULT_MATRIX_FORMAT):
	# creates S matrices for the PML creation

	M = np.prod(N)
	if np.isscalar(Npml): Npml = np.array([Npml])
	if len(N) < 2:
		N = append(N,1)
		Npml = append(Npml,0)
	Nx = N[0]
	Nx_pml = Npml[0]
	Ny = N[1]
	Ny_pml = Npml[1]

	# Create the sfactor in each direction and for 'f' and 'b'
	s_vector_x_f = create_sfactor(xrange, 'f', omega, Nx, Nx_pml)
	s_vector_x_b = create_sfactor(xrange, 'b', omega, Nx, Nx_pml)
	s_vector_y_f = create_sfactor(yrange, 'f', omega, Ny, Ny_pml)
	s_vector_y_b = create_sfactor(yrange, 'b', omega, Ny, Ny_pml)

	# Fill the 2D space with layers of appropriate s-factors
	Sx_f_2D = np.zeros(N, dtype=np.complex128)
	Sx_b_2D = np.zeros(N, dtype=np.complex128)
	Sy_f_2D = np.zeros(N, dtype=np.complex128)
	Sy_b_2D = np.zeros(N, dtype=np.complex128)

	for i in range(0, Nx):
		Sy_f_2D[:, i] = 1/s_vector_y_f
		Sy_b_2D[:, i] = 1/s_vector_y_b

	for j in range(0, Ny):
		Sx_f_2D[j, :] = 1/s_vector_x_f
		Sx_b_2D[j, :] = 1/s_vector_x_b

	# Reshape the 2D s-factors into a 1D s-array
	Sx_f_vec = Sx_f_2D.reshape((-1,))
	Sx_b_vec = Sx_b_2D.reshape((-1,))
	Sy_f_vec = Sy_f_2D.reshape((-1,))
	Sy_b_vec = Sy_b_2D.reshape((-1,))

	# Construct the 1D total s-array into a diagonal matrix
	Sx_f = sp.spdiags(Sx_f_vec, 0, M, M, format=matrix_format)
	Sx_b = sp.spdiags(Sx_b_vec, 0, M, M, format=matrix_format)
	Sy_f = sp.spdiags(Sy_f_vec, 0, M, M, format=matrix_format)
	Sy_b = sp.spdiags(Sy_b_vec, 0, M, M, format=matrix_format)

	return (Sx_f, Sx_b, Sy_f, Sy_b)


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
		pSolve = pardisoSolver(A, mtype=13) # Set matrix to complex unsymmetric
		pSolve.run_pardiso(12) # Factorize
		x = pSolve.run_pardiso(33, b) # Solve
		pSolve.clear()

	elif solver.lower() == 'scipy':
		x = spl.spsolve(A, b)

	else:
		raise ValueError('Invalid solver choice: {}, options are pardiso or scipy'.format(str(solver)))

	if timing:
		print('Linear system solve took {:.2f} seconds'.format(time()-t))

	return x

