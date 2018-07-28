# fdfdpy_OO

This is an object oriented version of [fdfdpy](https://github.com/fancompute/fdfdpy).

## Structure

The `Fdfd` class is initialized as

	FDFD = Fdfd(omega, dl, xrange, yrange, NPML, pol)

- `omega` : the angular frequency in units of $2\pi/s$
- `dl` : the spatial grid size in units of $m$
- `xrange` : defines spatial domain in x [left-most position, right-most position] in units of $m$
- `yrange` : defines spatial domain in y [bottom-most position, top-most position] in units of $m$
- `NPML` : defines number of PML grids [N on x border, N on y border]
- `pol` : polarization, one of {'Hz','Ez'} where $\hat{z}$ is the transverse field.

creating a new Fdfd object solves for the number of grid points in x and y, which are stored as `Nx` and `Ny` respectively.

also, it creates the relative permittiity and relative permeability arrays `eps_r` and `mu_r` as `numpy.ones((Nx,Ny))`

