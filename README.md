# fdfdpy_OO

This is an object oriented version of [fdfdpy](https://github.com/fancompute/fdfdpy).

## Structure

### Initialization

The `Fdfd` class is initialized as

	FDFD = Fdfd(omega, eps_r, dl, NPML, pol)

- `omega` : the angular frequency in units of $2\pi/s$
- `eps_r` : a numpy array specifying the relative permittivity distribution
- `dl` : the spatial grid size in units of $m$
- `NPML` : defines number of PML grids [# on x borders, # on y borders]
- `pol` : polarization, one of {'Hz','Ez'} where $\hat{z}$ is the transverse field.

Creating a new Fdfd object solves for:

- `xrange` : defines spatial domain in x [left-most position, right-most position] in units of $m$
- `yrange` : defines spatial domain in y [bottom-most position, top-most position] in units of $m$
- `A` : the Maxwell operator, which is used later to solve for the E&M fields.
- `derivs` : dictionary storing the derivative operators.

It also creates a `mu_r` as `numpy.ones(eps_r.shape)`. 

### Solving for the electromagnetic fields

Now, we have everything we need to solve the system for the electromagnetic fields, by running

	fields = FDFD.solve_fields(b)
	
`b` is proportional to either the `Jz` or `Mz` source term, depending on whether `pol` is set to `'Ez'` or `'Hz'`, respectively.  PLEASE NOTE: `b` is exacly the source for `Ax = b`, it is not a current density!.  `b.shape` must be `(Nx,Ny)`.

`fields` is a tuple containing `(Ex, Ey, Hz)` or `(Hx, Hy, Ez)` depending on the polarization.


### To Do

- [ ] Normalize the `A` matrix.
- [ ] Double check maxwell's equations for TM and TE field constructions.
- [ ] Save the factorization of `A` in the `Fdfd` object to be reused later if one has the same `A` but a different `b`.
- [ ] Allow the source term to have `(Jx, Jy, Jz, Mx, My, Mz)`, which would be useful for adjoint stuff where the source is not necessarily along the `z` direction.
- [ ] Allow for nonlinear A, where an electric field may be supplied when solving for the fields.
- [ ] Parallel sparse matrix solvers
- [ ] Add ability to run local jupyter notebooks running FDFD on parallel from hera.

