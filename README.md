# fdfdpy_OO

This is an object oriented version of [fdfdpy](https://github.com/fancompute/fdfdpy).

## Structure

### Initialization

The `Fdfd` class is initialized as

	FDFD = Fdfd(omega, dl, xrange, yrange, NPML, pol)

- `omega` : the angular frequency in units of $2\pi/s$
- `dl` : the spatial grid size in units of $m$
- `xrange` : defines spatial domain in x [left-most position, right-most position] in units of $m$
- `yrange` : defines spatial domain in y [bottom-most position, top-most position] in units of $m$
- `NPML` : defines number of PML grids [# on x borders, # on y borders]
- `pol` : polarization, one of {'Hz','Ez'} where $\hat{z}$ is the transverse field.

Creating a new Fdfd object solves for the number of grid points in x and y, which are stored as `Nx` and `Ny` respectively.

Also, it creates the relative permittiity and relative permeability arrays `eps_r` and `mu_r` as `numpy.ones((Nx,Ny))`.  These are meant to be reset later to suit the specific environment you wish to simulate, for example `FDFD.eps_r = new_eps`.

### Getting the system matrix

With the `Fdfd` object initialized (and your permittivity set), we may compute the maxwell operator `A` using the method

	A = FDFD.get_A()

This stores `A` in the Fdfd object and optionally returns it to the user.  `A` is a sparse matrix.

### Solving for the electromagnetic fields

Now, we have everything we need to solve the system for the electromagnetic fields, by running

	fields = FDFD.solve_fields(b)
	
`b` is either the `Jz` or `Mz` source term, depending on whether `pol` is set to `'Ez'` or `'Hz'`, respectively.  `b.shape` must be `(Nx,Ny)`.

`fields` is a tuple containing `(Ex, Ey, Ez, Hx, Hy, Hz)`.  Again, depending on the polarization, three out of these terms will be set to `None`.


### To Do

- Save the factorization of `A` in the `Fdfd` object to be reused later if one has the same `A` but a different `b`.
- Allow the source term to have `(Jx, Jy, Jz, Mx, My, Mz)`, which would be useful for adjoint stuff where the source is not necessarily along the `z` direction.
- Allow for nonlinear A, where an electric field may be supplied when solving for the fields.


