import numpy as np
from FDFD.linalg import construct_A, solver_direct, solver_eigs, unpack_derivs
from FDFD.constants import *

class Fdfd:

	def __init__(self, omega, eps_r, dl, NPML, pol):
		# initializes Fdfd object

		self.omega = float(omega)
		self.dl = float(dl)
		self.eps_r = eps_r
		self.NPML = [int(n) for n in NPML]
		self.pol = pol
		
		self._check_inputs()
		
		(Nx,Ny) = eps_r.shape
		self.Nx = Nx
		self.Ny = Ny
		self.mu_r = np.ones((self.Nx,self.Ny))
		self.xrange = [0, float(Nx*self.dl)]
		self.yrange = [0, float(Ny*self.dl)]
		
		# construct the system matrix
		(A, derivs) = construct_A(self.omega, self.xrange, self.yrange, eps_r, self.NPML, self.pol,
								matrix_format='csc', 
								timing=False)
		self.A = A
		self.derivs = derivs
		self.fields = {f : None for f in ['Ex','Ey','Ez','Hx','Hy','Hz']}


	def reset_eps(self, new_eps):
		# sets a new permittivity with the same other parameters and reconstructs a new A

		self.eps_r = new_eps
		(A, derivs) = construct_A(self.omega, self.xrange, self.yrange, self.eps_r, self.NPML, self.pol,
								matrix_format='csc', 
								timing=False)
		self.A = A
		self.derivs = derivs
		self.fields = {f : None for f in ['Ex','Ey','Ez','Hx','Hy','Hz']}


	def solve_fields(self, b, timing=False, solver='pardiso'):
		# performs direct solve for A given source b (note, b is not a current, it's literally the b in Ax = b)

		X = solver_direct(self.A, b, timing=timing, solver=solver)

		(Nx,Ny) = b.shape
		(Dyb, Dxb, Dxf, Dyf) = unpack_derivs(self.derivs)	

		if self.pol == 'Hz':
			ex = -1/1j/self.omega/EPSILON_0 * Dyb.dot(X)
			ey =  1/1j/self.omega/EPSILON_0 * Dxb.dot(X)

			Ex = ex.reshape((Nx, Ny), order='F')
			Ey = ey.reshape((Nx, Ny), order='F')
			Hz = X.reshape((Nx, Ny), order='F')

			self.derivs['Ex'] = Ex
			self.derivs['Ey'] = Ey
			self.derivs['Hz'] = Hz

			return (Ex, Ey ,Hz)

		elif self.pol == 'Ez':
			hx = -1/1j/self.omega/MU_0 * Dyb.dot(X)
			hy =  1/1j/self.omega/MU_0 * Dxb.dot(X)

			Hx = hx.reshape((Nx, Ny), order='F')
			Hy = hy.reshape((Nx, Ny), order='F')
			Ez = X.reshape((Nx, Ny), order='F')

			self.derivs['Hx'] = Hx
			self.derivs['Hy'] = Hy
			self.derivs['Ez'] = Ez

			return (Hx, Hy, Ez)

		else:
			raise ValueError('Invalid polarization: {}'.format(str(self.pol)))


	def _check_inputs(self):
		# checks the inputs and makes sure they are kosher
		
		assert len(self.NPML) == 2, "yrange must be a list of length 2, was supplied {}, which is of length {}".format(str(self.NPML), len(self.NPML))
		assert self.NPML[0] >= 0 and self.NPML[1] >= 0, "both elements of NPML must be >= 0"
		
		assert self.pol in ['Ez','Hz'], "pol must be one of 'Ez' or 'Hz'"
		
		# to do, check for correct types as well.
