from scipy.sparse import diags, kron, eye
from fdfdpy.constants import *

def createDws(w, s, dL, N, matrix_format=DEFAULT_MATRIX_FORMAT):
	# creates the derivative matrices
	# NOTE: python uses C ordering rather than Fortran ordering. Therefore the
	# derivative operators are constructed slightly differently than in MATLAB

	Nx = N[0]
	dx = dL[0]
	if len(N) is not 1:
		Ny = N[1]
		dy = dL[1]
	else:
		Ny = 1
		dy = np.inf
	if w is 'x':
		if s is 'f':
			dxf = diags([-1, 1, 1], [0, 1, -Nx+1], shape=(Nx, Nx))
			Dws = 1/dx*kron(dxf, eye(Ny), format=matrix_format)
		else:
			dxb = diags([1, -1, -1], [0, -1, Nx-1], shape=(Nx, Nx))
			Dws = 1/dx*kron(dxb, eye(Ny), format=matrix_format)
	if w is 'y':
		if s is 'f':
			dyf = diags([-1, 1, 1], [0, 1, -Ny+1], shape=(Ny, Ny))
			Dws = 1/dy*kron(eye(Nx), dyf, format=matrix_format)
		else:
			dyb = diags([1, -1, -1], [0, -1, Ny-1], shape=(Ny, Ny))
			Dws = 1/dy*kron(eye(Nx), dyb, format=matrix_format)
	return Dws


def unpack_derivs(derivs):
	# takes derivs dictionary and returns tuple for convenience

	Dyb = derivs['Dyb']
	Dxb = derivs['Dxb']
	Dxf = derivs['Dxf']
	Dyf = derivs['Dyf']
	return (Dyb, Dxb, Dxf, Dyf)
