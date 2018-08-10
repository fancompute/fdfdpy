import numpy as np
import scipy.sparse as sp

import copy

from fdfdpy.linalg import solver_direct, grid_average, solver_complex2real
from fdfdpy.derivatives import unpack_derivs
from fdfdpy.constants import *

# Note: for both solvers, the simulation object must have been initialized with the linear permittivity eps_r

def born_solve(simulation, nonlinear_fn, nl_region, Estart=None, conv_threshold=1e-10, max_num_iter=50, averaging=True):
	# solves for the nonlinear fields using direct substitution / Born approximation / Picard / whatever you want to call it

	# Stores convergence parameters
	conv_array = np.zeros((max_num_iter, 1))

	eps_lin = simulation.eps_r

	if simulation.pol == 'Ez':
		# Defne the starting field for the simulation
		if Estart is None:
			(Hx,Hy,Ez) = simulation.solve_fields()
		else: 
			Ez = Estart['Ez']	

		# Solve iteratively
		for istep in range(max_num_iter):

			Eprev = Ez

			# set new permittivity
			eps_nl = eps_lin + nonlinear_fn(Eprev)*nl_region

			# get new fields
			simulation.reset_eps(eps_nl)
			(Hx, Hy, Ez) = simulation.solve_fields()

			# get convergence and break
			convergence = np.linalg.norm(Ez - Eprev)/np.linalg.norm(Ez)
			conv_array[istep] = convergence

			# if below threshold, break and return
			if convergence < conv_threshold:
				break

		if convergence > conv_threshold:
			print("the simulation did not converge, reached {}".format(convergence))
		return (Hx, Hy, Ez, conv_array)

	elif simulation.pol == 'Hz':
		# Defne the starting field for the simulation
		if Estart is None:
			(Ex,Ey,Hz) = simulation.solve_fields(averaging=averaging)
		else: 
			Ex = Estart['Ex']
			Ey = Estart['Ey']		

		# Solve iteratively
		for istep in range(max_num_iter):

			Exprev = Ex
			Eyprev = Ey

			# set new permittivity
			eps_nl = eps_lin + (nonlinear_fn(Exprev) + nonlinear_fn(Eyprev))*nl_region 

			# get new fields
			simulation.reset_eps(eps_nl)
			(Ex, Ey, Hz) = simulation.solve_fields(averaging=averaging)

			# get convergence and break
			convergence = np.linalg.norm(Ex - Exprev)/np.linalg.norm(Ex) + np.linalg.norm(Ey - Eyprev)/np.linalg.norm(Ey)
			conv_array[istep] = convergence

			# if below threshold, break and return
			if convergence < conv_threshold:
				break

		if convergence > conv_threshold:
			print("the simulation did not converge, reached {}".format(convergence))
		return (Ex, Ey, Hz, conv_array)


def newton_solve(simulation, nonlinear_fn, nl_region, nonlinear_de, 
				Estart=None, conv_threshold=1e-10, max_num_iter=50, averaging=True,
				solver=DEFAULT_SOLVER, jac_solver='c2r', matrix_format=DEFAULT_MATRIX_FORMAT):
	# solves for the nonlinear fields using Newton's method

	eps_lin = simulation.eps_r
	b = simulation.src

	# Stores convergence parameters
	conv_array = np.zeros((max_num_iter, 1))

	# num. columns and rows of A
	Nbig = simulation.Nx*simulation.Ny

	if simulation.pol == 'Ez':
		# Defne the starting field for the simulation
		if Estart is None:
			(Hx,Hy,Ez) = simulation.solve_fields()
		else: 
			Ez = Estart

		# Solve iteratively
		for istep in range(max_num_iter):

			Eprev = Ez

			(fx, Jac11, Jac12) = nl_eq_and_jac(simulation, b, eps_lin, nonlinear_fn, nl_region, nonlinear_de, 
											Ez=Eprev, matrix_format=matrix_format)

			# Note: Newton's method is defined as a linear problem to avoid inverting the Jacobian
			# Namely, J*(x_n - x_{n-1}) = -f(x_{n-1}), where J = df/dx(x_{n-1})

			Ediff = solver_complex2real(Jac11, Jac12, fx, solver=solver, timing=False)
			Ez = Eprev - Ediff[range(Nbig)].reshape(simulation.Nx, simulation.Ny)

			# get convergence and break
			convergence = np.linalg.norm(Ez - Eprev)/np.linalg.norm(Ez)
			conv_array[istep] = convergence

			# if below threshold, break and return
			if convergence < conv_threshold:
				break

		# Solve the fdfd problem with the final eps_nl
		eps_nl = eps_lin + (nonlinear_fn(Ez)*nl_region)
		simulation.reset_eps(eps_nl)
		(Hx, Hy, Ez) = simulation.solve_fields()

		if convergence > conv_threshold:
			print("the simulation did not converge, reached {}".format(convergence))
			
		return (Hx, Hy, Ez, conv_array)

	elif simulation.pol == 'Hz':
		# Defne the starting field for the simulation
		if Estart is None:
			(Ex,Ey,Hz) = simulation.solve_fields(averaging=averaging)
		else: 
			Ex = Estart['Ex']
			Ey = Estart['Ey']

		# Solve iteratively
		for istep in range(max_num_iter):

			Exprev = Ex
			Eyprev = Ey

			(fx, Jac11, Jac12) = nl_eq_and_jac(simulation, b, eps_lin, nonlinear_fn, nl_region, nonlinear_de, 
											Ex=Exprev, Ey=Eyprev, matrix_format=matrix_format, averaging=averaging)

			# Note: Newton's method is defined as a linear problem to avoid inverting the Jacobian
			# Namely, J*(x_n - x_{n-1}) = -f(x_{n-1}), where J = df/dx(x_{n-1})

			Ediff = solver_complex2real(Jac11, Jac12, fx, solver=solver, timing=False)

			Ex = Exprev - Ediff[range(Nbig)].reshape(simulation.Nx, simulation.Ny)
			Ey = Eyprev - Ediff[range(Nbig, 2*Nbig)].reshape(simulation.Nx, simulation.Ny)

			# get convergence and break
			convergence = np.linalg.norm(Ex - Exprev)/np.linalg.norm(Ex) + np.linalg.norm(Ey - Eyprev)/np.linalg.norm(Ey)
			conv_array[istep] = convergence

			# if below threshold, break and return
			if convergence < conv_threshold:
				break

		# Solve the fdfd problem with the final eps_nl
		eps_nl = eps_lin + (nonlinear_fn(Exprev) + nonlinear_fn(Eyprev))*nl_region 
		simulation.reset_eps(eps_nl)
		(Hx, Hy, Ez) = simulation.solve_fields(averaging=averaging)

		if convergence > conv_threshold:
			print("the simulation did not converge, reached {}".format(convergence))
			
		return (Ex, Ey, Hz, conv_array)

def nl_eq_and_jac(simulation, b, eps_lin, nonlinear_fn, nl_region, nonlinear_de, averaging=True,
				Ex=None, Ey=None, Ez=None, matrix_format=DEFAULT_MATRIX_FORMAT):
	# Evaluates the nonlinear function f(E) that defines the problem to solve f(E) = 0, as well as the Jacobian df/dE
	# Could add a check that only Ez is None for Hz polarization and vice-versa

	omega = simulation.omega
	EPSILON_0_ = EPSILON_0*simulation.L0
	MU_0_ = MU_0*simulation.L0

	Nbig = simulation.Nx*simulation.Ny
	
	# Set nonlinear permittivity
	eps_nl = eps_lin + sum(nonlinear_fn(e) for e in (Ex, Ey, Ez) if e is not None)*nl_region

	# Reset simulation for matrix A 
	simulation.reset_eps(eps_nl)

	if simulation.pol == 'Ez':

		Anl = simulation.A 
		fE = (Anl.dot(Ez.reshape(-1,)) - b.reshape(-1,)*1j*omega)

		# Make it explicitly a column vector
		fE = fE.reshape(Nbig, 1)

		dAde = (nonlinear_de(Ez)*nl_region).reshape((-1,))*omega**2*EPSILON_0_ 
		Jac11 = Anl + sp.spdiags(dAde*Ez.reshape((-1,)), 0, Nbig, Nbig, format=matrix_format)
		Jac12 = sp.spdiags(np.conj(dAde)*Ez.reshape((-1,)), 0, Nbig, Nbig, format=matrix_format)

	elif simulation.pol == 'Hz':

		(Dyb, Dxb, Dxf, Dyf) = unpack_derivs(simulation.derivs)	
		if averaging:
			vector_eps_x = grid_average(EPSILON_0_*eps_nl, 'x').reshape((Nbig,1))
			vector_eps_y = grid_average(EPSILON_0_*eps_nl, 'y').reshape((Nbig,1))
		else:
			vector_eps_x = EPSILON_0_*eps_nl.reshape((Nbig,1))
			vector_eps_y = EPSILON_0_*eps_nl.reshape((Nbig,1))

		Axx = -Dyb.dot(Dyf)/MU_0_
		Axy = Dyb.dot(Dxf)/MU_0_
		Ayx = Dxb.dot(Dyf)/MU_0_
		Ayy = -Dxb.dot(Dxf)/MU_0_
		Anl = sp.vstack((sp.hstack((Axx, Axy)), (sp.hstack((Ayx, Ayy)))))

		Exy = np.vstack((Ex.reshape((Nbig,1)), Ey.reshape((Nbig,1))))
		eps_xy = np.vstack((vector_eps_x, vector_eps_y))

		Anl = Anl - omega**2*sp.spdiags(eps_xy.reshape((-1,)), 0, 2*Nbig, 2*Nbig, format=matrix_format) 
		fE = Anl.dot(Exy) - np.vstack((Dyb.dot(b.reshape((Nbig, 1))), -Dxb.dot(b.reshape((Nbig, 1)))))/MU_0_

		if averaging:
			dAdex = -grid_average(nonlinear_de(Ex)*nl_region, 'x').reshape((-1,))*omega**2*EPSILON_0_ 
			dAdey = -grid_average(nonlinear_de(Ey)*nl_region, 'y').reshape((-1,))*omega**2*EPSILON_0_ 
		else:
			dAdex = -(nonlinear_de(Ex)*nl_region).reshape((-1,))*omega**2*EPSILON_0_ 
			dAdey = -(nonlinear_de(Ey)*nl_region).reshape((-1,))*omega**2*EPSILON_0_ 

		Jac11xx = sp.spdiags(dAdex*Ex.reshape((-1,)), 0, Nbig, Nbig, format=matrix_format)
		Jac11xy = sp.spdiags(dAdex*Ey.reshape((-1,)), 0, Nbig, Nbig, format=matrix_format)
		Jac11yx = sp.spdiags(dAdey*Ex.reshape((-1,)), 0, Nbig, Nbig, format=matrix_format)
		Jac11yy = sp.spdiags(dAdey*Ey.reshape((-1,)), 0, Nbig, Nbig, format=matrix_format)

		Jac12xx = sp.spdiags(np.conj(dAdex)*Ex.reshape((-1,)), 0, Nbig, Nbig, format=matrix_format)
		Jac12xy = sp.spdiags(np.conj(dAdex)*Ey.reshape((-1,)), 0, Nbig, Nbig, format=matrix_format)
		Jac12yx = sp.spdiags(np.conj(dAdey)*Ex.reshape((-1,)), 0, Nbig, Nbig, format=matrix_format)
		Jac12yy = sp.spdiags(np.conj(dAdey)*Ey.reshape((-1,)), 0, Nbig, Nbig, format=matrix_format)

		Jac11 = Anl + sp.vstack((sp.hstack((Jac11xx, Jac11xy)), sp.hstack((Jac11yx, Jac11yy))))
		Jac12 = sp.vstack((sp.hstack((Jac12xx, Jac12xy)), sp.hstack((Jac12yx, Jac12yy))))

	return(fE, Jac11, Jac12)
