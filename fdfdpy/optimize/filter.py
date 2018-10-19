import matplotlib.pylab as plt
import numpy as np
import autograd.numpy as npa
import scipy.sparse as sp

from autograd import grad
from functools import partial

# eps = np.load('data/figs/data/2port_eps.npy')


def dist(r1, r2):
    return np.sqrt(np.sum(np.square(r1 - r2)))

def sub2ind(array_shape, rows, cols):
    return rows*array_shape[1] + cols

def get_W(Nx, Ny, design_region, R=10):

    diffs = range(-R, R+1)
    N = Nx*Ny

    row_indeces = []
    col_indeces = []
    vals = []

    for i1 in range(Nx):
        for j1 in range(Ny):
            r1 = np.array([i1, j1])

            row_index = sub2ind((Nx, Ny), i1, j1)

            if i1 <= R or i1 >= Nx-R-1:
                pass
                # row_indeces.append(row_index)
                # col_indeces.append(row_index)
                # vals.append(R)
            elif j1 <= R or j1 >= Ny-R-1:
                pass
                # row_indeces.append(row_index)
                # col_indeces.append(row_index)
                # vals.append(R)
            else:
                for i_diff in diffs:
                    i2 = i1 + i_diff
                    for j_diff in diffs:
                        j2 = j1 + j_diff
                        r2 = np.array([i2, j2])
                        col_index = sub2ind((Nx, Ny), i2, j2)

                        val = R - dist(r1, r2)

                        if val > 0:
                            row_indeces.append(row_index)
                            col_indeces.append(col_index)
                            vals.append(val)

    W = sp.csr_matrix((vals, (row_indeces, col_indeces)), shape=(N, N))

    des_vec = design_region.reshape((-1,))
    no_des_vec = 1-des_vec
    des_mat = sp.diags(des_vec, shape=(N, N))
    no_des_mat = sp.diags(no_des_vec, shape=(N, N))

    W_des = W.dot(des_mat) + no_des_mat

    norm_vec = W.dot(np.ones((Nx*Ny,)))
    norm_vec[norm_vec == 0] = 1
    norm_mat = sp.diags(1/norm_vec, shape=(N, N))

    W = W.dot(norm_mat)

    return des_mat.dot(W) + no_des_mat


""" THESE ARE THE OPERATIONS GOING FROM A DENSITY TO A PERMITTIVITY
    THROUGH FILTERING AND PROJECTING """


def rho2rhot(rho, W):
    # density to filtered density
    (Nx, Ny) = rho.shape
    rho_vec = rho.reshape((-1,))
    rhot_vec = W.dot(rho_vec)
    rhot = np.reshape(rhot_vec, (Nx, Ny))
    return rhot


def rhot2rhob(rhot, eta=0.5, beta=100):
    # filtered density to projected density
    num = np.tanh(beta*eta) + np.tanh(beta*(rhot - eta))
    den = np.tanh(beta*eta) + np.tanh(beta*(1 - eta))
    return num / den


def rhob2eps(rhob, eps_m):
    # filtered density to permittivity
    return 1 + rhob * (eps_m - 1)


def eps2rho(eps, eps_m):
    # permittivity to density (ONLY USED FOR STARTING THE SIMULATION!)
    return (eps - 1) / (eps_m - 1)


def rho2eps(rho, eps_m, W, eta=0.5, beta=100):
    rhot = rho2rhot(rho, W)
    rhob = rhot2rhob(rhot, eta=eta, beta=beta)
    eps = rhob2eps(rhob, eps_m=eps_m)
    return eps


"""" DERIVATIVE OPERATORS """


def drhot_drho(W):
    # derivative of filtered density with respect to design density
    return W


def drhob_drhot(rho_t, eta=0.5, beta=100):
    # change in projected density with respect to filtered density
    rhot_vec = np.reshape(rho_t, (-1,))
    num = beta - beta*np.square(np.tanh(beta*(rhot_vec - eta)))
    den = np.tanh(beta*eta) + np.tanh(beta*(1 - eta))
    return num / den

def deps_drhob(rhob, eps_m):
    # change in permittivity with respect to projected density
    return (eps_m - 1)
