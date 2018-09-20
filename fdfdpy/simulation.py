import numpy as np
import scipy.sparse as sp
from copy import deepcopy

from fdfdpy.linalg import construct_A, solver_direct, grid_average
from fdfdpy.derivatives import unpack_derivs
from fdfdpy.plot import plt_base, plt_base_eps
from fdfdpy.nonlinear_solvers import born_solve, newton_solve, LM_solve
from fdfdpy.source.mode import mode
from fdfdpy.constants import (DEFAULT_LENGTH_SCALE, DEFAULT_MATRIX_FORMAT,
                              DEFAULT_SOLVER, EPSILON_0, MU_0)


class Simulation:

    def __init__(self, omega, eps_r, dl, NPML, pol, L0=DEFAULT_LENGTH_SCALE):
        # initializes Fdfd object

        self.L0 = L0
        self.omega = float(omega)
        self.dl = float(dl)
        self.NPML = [int(n) for n in NPML]
        self.pol = pol

        self._check_inputs()

        (Nx, Ny) = eps_r.shape
        self.Nx = Nx
        self.Ny = Ny
        self.mu_r = np.ones((self.Nx, self.Ny))
        self.src = np.zeros((self.Nx, self.Ny))
        self.xrange = [0, float(Nx*self.dl)]
        self.yrange = [0, float(Ny*self.dl)]

        # construct the system matrix
        self.eps_r = eps_r
        self.modes = []

    def setup_modes(self):
        # calculates
        for modei in self.modes:
            modei.setup_src(self)

    def add_mode(self, neff, direction_normal, center, width,
                 scale=1, order=1):
        # adds a mode definition to the simulation
        new_mode = mode(neff, direction_normal, center, width,
                        scale=scale, order=order)
        self.modes.append(new_mode)

    @property
    def eps_r(self):
        return self.__eps_r

    @eps_r.setter
    def eps_r(self, eps_r):
        self.__eps_r = eps_r
        (A, derivs) = construct_A(self.omega, self.xrange, self.yrange,
                                  self.__eps_r, self.NPML, self.pol, self.L0,
                                  matrix_format=DEFAULT_MATRIX_FORMAT,
                                  timing=False)
        self.A = A
        self.derivs = derivs
        self.fields = {f: None for f in ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']}

    def reset_eps(self, new_eps):
        # in here for compatibility for now..

        self.eps_r = new_eps
        (A, derivs) = construct_A(self.omega, self.xrange, self.yrange, self.eps_r, self.NPML, self.pol, self.L0,
                                matrix_format=DEFAULT_MATRIX_FORMAT,
                                timing=False)
        self.A = A
        self.derivs = derivs
        self.fields = {f: None for f in ['Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz']}

    def solve_fields(self, timing=False, averaging=True, solver=DEFAULT_SOLVER,
                     matrix_format=DEFAULT_MATRIX_FORMAT):
        # performs direct solve for A given source

        EPSILON_0_ = EPSILON_0*self.L0
        MU_0_ = MU_0*self.L0

        X = solver_direct(self.A, self.src*1j*self.omega, timing=timing,
                          solver=solver)

        (Nx, Ny) = self.src.shape
        M = Nx*Ny
        (Dyb, Dxb, Dxf, Dyf) = unpack_derivs(self.derivs)

        if self.pol == 'Hz':
            if averaging:
                eps_x = grid_average(EPSILON_0_*self.eps_r, 'x')
                vector_eps_x = eps_x.reshape((-1,))
                eps_y = grid_average(EPSILON_0_*self.eps_r, 'y')
                vector_eps_y = eps_y.reshape((-1,))
            else:
                vector_eps_x = EPSILON_0_*self.eps_r.reshape((-1,))
                vector_eps_y = EPSILON_0_*self.eps_r.reshape((-1,))

            T_eps_x_inv = sp.spdiags(1/vector_eps_x, 0, M, M,
                                  format=matrix_format)
            T_eps_y_inv = sp.spdiags(1/vector_eps_y, 0, M, M,
                                  format=matrix_format)

            ex = 1/1j/self.omega * T_eps_y_inv.dot(Dyb).dot(X)
            ey = -1/1j/self.omega * T_eps_x_inv.dot(Dxb).dot(X)

            Ex = ex.reshape((Nx, Ny))
            Ey = ey.reshape((Nx, Ny))
            Hz = X.reshape((Nx, Ny))

            self.fields['Ex'] = Ex
            self.fields['Ey'] = Ey
            self.fields['Hz'] = Hz

            return (Ex, Ey, Hz)

        elif self.pol == 'Ez':
            hx = -1/1j/self.omega/MU_0_ * Dyb.dot(X)
            hy = 1/1j/self.omega/MU_0_ * Dxb.dot(X)

            Hx = hx.reshape((Nx, Ny))
            Hy = hy.reshape((Nx, Ny))
            Ez = X.reshape((Nx, Ny))

            self.fields['Hx'] = Hx
            self.fields['Hy'] = Hy
            self.fields['Ez'] = Ez

            return (Hx, Hy, Ez)

        else:
            raise ValueError('Invalid polarization: {}'.format(str(self.pol)))

    def solve_fields_nl(self, nonlinear_fn, nl_region, dnl_de=None,
                        timing=False, averaging=True,
                        Estart=None, solver_nl='born', conv_threshold=1e-10,
                        max_num_iter=50, solver=DEFAULT_SOLVER,
                        matrix_format=DEFAULT_MATRIX_FORMAT):
        # solves for the nonlinear fields of the simulation.

        # store the original permittivity
        eps_orig = deepcopy(self.eps_r)

        # if the nonlinear objects were not supplied, throw an error
        if nonlinear_fn is None or nl_region is None:
            raise ValueError("'nonlinear_fn' and 'nl_region' must be supplied")

        if self.pol == 'Ez':
            # if born solver
            if solver_nl == 'born':

                (Hx, Hy, Ez, conv_array) = born_solve(self, nonlinear_fn,
                                                      nl_region, Estart,
                                                      conv_threshold,
                                                      max_num_iter,
                                                      averaging=averaging)

            # if newton solver
            elif solver_nl == 'newton':

                # newton needs the derivative of the nonlinearity.
                if dnl_de is None:
                    raise ValueError("'dnl_de' argument must be set to run"
                                     " Newton solve")

                (Hx, Hy, Ez, conv_array) = newton_solve(self, nonlinear_fn,
                                                        nl_region, dnl_de,
                                                        Estart, conv_threshold,
                                                        max_num_iter,
                                                        averaging=averaging)

            elif solver_nl == 'LM':

                # LM needs the derivative of the nonlinearity.
                if dnl_de is None:
                    raise ValueError("'dnl_de' argument must be set to run"
                                     " LM solve")

                (Hx, Hy, Ez, conv_array) = LM_solve(self, nonlinear_fn,
                                                    nl_region, dnl_de, Estart,
                                                    conv_threshold,
                                                    max_num_iter,
                                                    averaging=averaging)

            # incorrect solver_nl argument
            else:
                raise AssertionError("solver must be one of "
                                     "{'born', 'newton', 'LM'}")

            # reset the permittivity to the original value
            self.reset_eps(eps_orig)

            # return final nonlinear fields and an array of the convergences

            self.fields['Hx'] = Hx
            self.fields['Hy'] = Hy
            self.fields['Ez'] = Ez

            return (Hx, Hy, Ez, conv_array)

        elif self.pol == 'Hz':
            # if born solver
            if solver_nl == 'born':

                (Ex, Ey, Hz, conv_array) = born_solve(self, nonlinear_fn,
                                                      nl_region, Estart,
                                                      conv_threshold,
                                                      max_num_iter,
                                                      averaging=averaging)

            # if newton solver
            elif solver_nl == 'newton':

                # newton needs the derivative of the nonlinearity.
                if dnl_de is None:
                    raise ValueError("'dnl_de' argument must be set to run "
                                     "Newton solve")
                (Ex, Ey, Hz, conv_array) = newton_solve(self, nonlinear_fn,
                                                        nl_region, dnl_de,
                                                        Estart, conv_threshold,
                                                        max_num_iter,
                                                        averaging=averaging)

            # incorrect solver_nl argument
            else:
                raise AssertionError("solver must be one of "
                                     "{'born', 'newton'}")

            # reset the permittivity to the original value
            # (note, not self.reset_eps or else the fields get destroyed)
            self.reset_eps(eps_orig)

            # return final nonlinear fields and an array of the convergences

            self.fields['Ex'] = Ex
            self.fields['Ey'] = Ey
            self.fields['Hz'] = Hz

            return (Ex, Ey, Hz, conv_array)

        else:
            raise ValueError('Invalid polarization: {}'.format(str(self.pol)))

    def _check_inputs(self):
        # checks the inputs and makes sure they are kosher

        assert self.L0 > 0, "L0 must be a positive number, was supplied {},".format(str(self.L0))
        assert len(self.NPML) == 2, "yrange must be a list of length 2, was supplied {}, which is of length {}".format(str(self.NPML), len(self.NPML))
        assert self.NPML[0] >= 0 and self.NPML[1] >= 0, "both elements of NPML must be >= 0"

        assert self.pol in ['Ez', 'Hz'], "pol must be one of 'Ez' or 'Hz'"

        # to do, check for correct types as well.

    def flux_probe(self, direction_normal, center, width):
        # computes the total flux across the plane (line in 2D) defined by direction_normal, center, width

        # first extract the slice of the permittivity
        if direction_normal == "x":
            inds_x = [center[0], center[0]+1]
            inds_y = [int(center[1]-width/2), int(center[1]+width/2)]
        elif direction_normal == "y":
            inds_x = [int(center[0]-width/2), int(center[0]+width/2)]
            inds_y = [center[1], center[1]+1]
        else:
            raise ValueError("The value of direction_normal is neither x nor y!")

        if self.pol == 'Ez':
            Ez_x = grid_average(self.fields['Ez'][inds_x[0]:inds_x[1]+1, inds_y[0]:inds_y[1]+1], 'x')[:-1,:-1]
            Ez_y = grid_average(self.fields['Ez'][inds_x[0]:inds_x[1]+1, inds_y[0]:inds_y[1]+1], 'y')[:-1,:-1]
            # NOTE: Last part drops the extra rows/cols used for grid_average

            if direction_normal == "x":
                Sx = -1/2*np.real(Ez_x*np.conj(self.fields['Hy'][inds_x[0]:inds_x[1], inds_y[0]:inds_y[1]]))
                return self.dl*np.sum(Sx)
            elif direction_normal == "y":
                Sy = 1/2*np.real(Ez_y*np.conj(self.fields['Hy'][inds_x[0]:inds_x[1], inds_y[0]:inds_y[1]]))
                return self.dl*np.sum(Sy)

        elif self.pol == 'Hz':
            Hz_x = grid_average(self.fields['Hz'][inds_x[0]:inds_x[1]+1, inds_y[0]:inds_y[1]+1], 'x')[:-1, :-1]
            Hz_y = grid_average(self.fields['Hz'][inds_x[0]:inds_x[1]+1, inds_y[0]:inds_y[1]+1], 'y')[:-1, :-1]
            # NOTE: Last part drops the extra rows/cols used for grid_average

            if direction_normal == "x":
                Sx = 1/2*np.real(self.fields['Ey'][inds_x[0]:inds_x[1], inds_y[0]:inds_y[1]]*np.conj(Hz_x))
                return self.dl*np.sum(Sx)
            elif direction_normal == "y":
                Sy = -1/2*np.real(self.fields['Ex'][inds_x[0]:inds_x[1], inds_y[0]:inds_y[1]]*np.conj(Hz_y))
                return self.dl*np.sum(Sy)

    def plt_abs(self, cbar=True, outline=True, ax=None):
        # plot np.absolute value of primary field (e.g. Ez/Hz)

        if self.fields[self.pol] is None:
            raise ValueError("need to solve the simulation first")

        field_val = np.abs(self.fields[self.pol])
        outline_val = np.abs(self.eps_r)
        vmin = 0.0
        vmax = field_val.max()
        cmap = "magma"

        return plt_base(field_val, outline_val, cmap, vmin, vmax, self.pol,
                        cbar=cbar, outline=outline, ax=ax)

    def plt_re(self, cbar=True, outline=True, ax=None):
        # plot np.real part of primary field (e.g. Ez/Hz)

        if self.fields[self.pol] is None:
            raise ValueError("need to solve the simulation first")

        field_val = np.real(self.fields[self.pol])
        outline_val = np.abs(self.eps_r)
        vmin = -np.abs(field_val).max()
        vmax = +np.abs(field_val).max()
        cmap = "RdBu"

        return plt_base(field_val, outline_val, cmap, vmin, vmax, self.pol,
                        cbar=cbar, outline=outline, ax=ax)

    def plt_eps(self, cbar=True, outline=True, ax=None):
        # plot the permittivity distribution

        eps_val = np.abs(self.eps_r)
        outline_val = np.abs(self.eps_r)
        vmin = 1
        vmax = np.abs(self.eps_r).max()
        cmap = "Greys"

        return plt_base_eps(eps_val, outline_val, cmap, vmin, vmax, cbar=cbar,
                            outline=outline, ax=ax)
