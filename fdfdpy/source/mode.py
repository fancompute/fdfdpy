import numpy as np
import scipy.sparse as sp

from fdfdpy.constants import *
from fdfdpy.linalg import *

class mode:
	def __init__(self, neff, direction_normal, center, width, order=1):
		self.neff = neff
		self.direction_normal = direction_normal
		self.center = center
		self.width = width
		self.order = order


	def setup_src(self, simulation, matrix_format=DEFAULT_MATRIX_FORMAT):
		self.insert_mode(simulation, simulation.src, matrix_format=matrix_format)


	def insert_mode(self, simulation, destination, matrix_format=DEFAULT_MATRIX_FORMAT):
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

		vector_eps = EPSILON_0_*eps_r.reshape((-1,))
		vector_eps_x = EPSILON_0_*grid_average(eps_r,'x').reshape((-1,))
		T_eps = sp.spdiags(vector_eps, 0, N, N, format=matrix_format)
		T_epsxinv = sp.spdiags(vector_eps_x**(-1), 0, N, N, format=matrix_format)

		if simulation.pol == 'Ez':
			A = np.square(simulation.omega)*MU_0_*T_eps + Dxf.dot(Dxb)

		elif simulation.pol == 'Hz':
			A = np.square(simulation.omega)*MU_0_*T_eps + T_eps.dot(Dxf).dot(T_epsxinv).dot(Dxb)

		est_beta = simulation.omega*np.sqrt(MU_0_*EPSILON_0_)*self.neff
		(vals, vecs) = solver_eigs(A, self.order, guess_value=np.square(est_beta))

		if self.order == 1:
			src = vecs
		else:
			print(vecs.shape)
			src = vecs[:,self.order-1]

		if self.direction_normal == 'x':
			src = src.reshape((1,-1))
			destination[ inds_x[0]:inds_x[1], inds_y[0]:inds_y[1] ] = np.abs(src)*np.sign(np.real(src))
		else:
			src = src.reshape((-1,1))
			destination[ inds_x[0]:inds_x[1], inds_y[0]:inds_y[1] ] = np.abs(src)*np.sign(np.real(src))
