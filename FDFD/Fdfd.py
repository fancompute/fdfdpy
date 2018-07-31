import numpy as np
from FDFD.linalg import construct_A, solver_direct, solver_eigs

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

	def solve_fields(self, b, timing=False, solver='pardiso.parts'):
		# performs direct solve for A given source b (note, b is not a current, it's literally the b in Ax = b)
		b = b.astype(np.complex128)

		(field_X,field_Y,field_Z) = solver_direct(self.A, b, self.derivs, self.omega, self.pol, timing=timing, solver=solver)

		if self.pol == 'Hz':
			self.derivs['Ex'] = field_X
			self.derivs['Ey'] = field_Y
			self.derivs['Hz'] = field_Z
		else:
			self.derivs['Hx'] = field_X
			self.derivs['Hy'] = field_Y
			self.derivs['Ez'] = field_Z

		return (field_X,field_Y,field_Z)


	def _check_inputs(self):
		# checks the inputs and makes sure they are kosher
		
		assert len(self.NPML) == 2, "yrange must be a list of length 2, was supplied {}, which is of length {}".format(str(self.NPML), len(self.NPML))
		assert self.NPML[0] >= 0 and self.NPML[1] >= 0, "both elements of NPML must be >= 0"
		
		assert self.pol in ['Ez','Hz'], "pol must be one of 'Ez' or 'Hz'"
		
		# to do, check for correct types as well.
