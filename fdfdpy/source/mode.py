import numpy as np
import scipy.sparse as sp

from fdfdpy.constants import *
from fdfdpy.linalg import *

class mode:
	def __init__(self, neff, direction_normal, center, width):
		self.neff = neff
		self.direction_normal = direction_normal
		self.center = center
		self.width = width

	def setup_src(self, simulation, matrix_format=DEFAULT_MATRIX_FORMAT):
		EPSILON_0_ = EPSILON_0*simulation.L0
		MU_0_ = MU_0*simulation.L0

		# first extract the slice of the permittivity 
		if self.direction_normal == "x":
			inds_x = [self.center[0], self.center[0]+1]
			inds_y = [int(self.center[1]-self.width/2), int(self.center[1]+self.width/2)]
		elif self.direction_normal == "y":
			inds_x = [int(self.center[0]-self.width/2), int(self.center[0]+self.width/2)]
			inds_y = [self.center[1], self.center[1]+1]
		else:
			raise ValueError("The value of direction_normal is not x or y!")

		eps_r = simulation.eps_r[ inds_x[0]:inds_x[1], inds_y[0]:inds_y[1] ]
		N = eps_r.size

		Dxb = createDws('x', 'b', [simulation.dl], [N], matrix_format=matrix_format)
		Dxf = createDws('x', 'f', [simulation.dl], [N], matrix_format=matrix_format)

		vector_eps_z = EPSILON_0_*eps_r.reshape((-1,))
		T_eps = sp.spdiags(vector_eps_z, 0, N, N, format=matrix_format)

		if simulation.pol == "Ez":
			A = np.square(simulation.omega)*MU_0_*T_eps + Dxf.dot(Dxb)

		elif simulation.pol == "Hz":
			A = np.square(simulation.omega)*MU_0_*T_eps + Dxf.dot(Dxb)
			# A = ω^2*μ₀*Tϵ + Tϵ*δxf*Tϵxinv*δxb

		est_beta = simulation.omega*np.sqrt(MU_0_*EPSILON_0_)*self.neff
		(vals, vecs) = solver_eigs(A, 1, guess_value=np.square(est_beta))

		simulation.src[ inds_x[0]:inds_x[1], inds_y[0]:inds_y[1] ] = np.real(vecs.reshape((-1,)))
	    