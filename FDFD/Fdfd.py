import numpy as np
from FDFD.linalg import construct_A, solver_direct, solver_eigs

class Fdfd:

	def __init__(self, omega, dl, xrange, yrange, NPML, pol):
		# initializes Fdfd object

		self.omega = float(omega)
		self.dl = float(dl)
		self.xrange = [float(x) for x in xrange]
		self.yrange = [float(y) for y in yrange]
		self.NPML = [int(n) for n in NPML]
		self.pol = pol
		
		self._check_inputs()
		
		self.Nx = int((xrange[1]-xrange[0])/dl)
		self.Ny = int((yrange[1]-yrange[0])/dl)
		self.eps_r = np.ones((self.Nx,self.Ny))
		self.mu_r = np.ones((self.Nx,self.Ny))
		self.A = None
		self.derivs = {}


	def create_A(self, eps_r=None):
		# makes operator and sets permittivity (optional)
		
		if eps_r is None:
			eps_r = self.eps_r
		
		# constructs A from the linalg file
		(A, derivs) = construct_A(self.omega, self.xrange, self.yrange, eps_r, self.NPML, self.pol,
						matrix_format='csc', 
						timing=False)

		self.A = A
		self.derivs = derivs
		
		return A


	def solve_fields(self, b):
		# performs direct solve for A given source b (note, b is not a current, it's literally the b in Ax = b)
		
		(field_X,field_Y,field_Z) = solver_direct(self.A, b, self.derivs, self.omega, self.pol, timing=False)

		return (field_X,field_Y,field_Z)


	def _check_inputs(self):
		# checks the inputs and makes sure they are kosher
		
		assert len(self.xrange) == 2, "xrange must be a list of length 2, was supplied {}, which is of length {}".format(str(self.xrange), len(self.xrange))
		assert self.xrange[1] > self.xrange[0], "second element of xrange must be greater than first element, was supplied {}".format(str(self.xrange))		

		assert len(self.xrange) == 2, "yrange must be a list of length 2, was supplied {}, which is of length {}".format(str(self.yrange), len(self.yrange))
		assert self.yrange[1] > self.yrange[0], "second element of yrange must be greater than first element, was supplied {}".format(str(self.yrange))

		assert len(self.NPML) == 2, "yrange must be a list of length 2, was supplied {}, which is of length {}".format(str(self.NPML), len(self.NPML))
		assert self.NPML[0] >= 0 and self.NPML[1] >= 0, "both elements of NPML must be >= 0"
		
		assert self.pol in ['Ez','Hz'], "pol must be one of 'Ez' or 'Hz'"
		
		# to do, check for correct types as well.