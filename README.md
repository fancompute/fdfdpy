![](img/dipole_dielectric_field.png)

# fdfdpy

This is a pure Python implementation of the finite difference frequency domain (FDFD) method. It makes use of scipy, numpy, matplotlib, and the MKL Pardiso solver. fdfdpy currently supports 2D geometries

## Installation

    python setup.py install

## Examples

See the ipython notebooks in `notebooks`.

## Unit Tests

Some basic tests are included in `tests/`

To run an example test, `tests/test_nonlinear_solvers.py`, either call

	python -m unittest tests/test_nonlinear_solvers.py

or

	python tests/test_nonlinear_solvers.py

## Structure

### Initialization

The `Simulation` class is initialized as

	from fdfdpy import Simulation
	simulation = Simulation(omega, eps_r, dl, NPML, pol, L0)

- `omega` : the angular frequency in units of` 2 pi / seconds`
- `eps_r` : a numpy array specifying the relative permittivity distribution
- `dl` : the spatial grid size in units of `L0`
- `NPML` : defines number of PML grids `[# on x borders, # on y borders]`
- `pol` : polarization, one of `{'Hz','Ez'}` where `z` is the transverse field.
- `L0` : simulation length scale, default is 1e-6 meters (one micron)

Creating a new Fdfd object solves for:

- `xrange` : defines spatial domain in x [left-most position, right-most position] in units of `L0`
- `yrange` : defines spatial domain in y [bottom-most position, top-most position] in units of `L0`
- `A` : the Maxwell operator, which is used later to solve for the E&M fields.
- `derivs` : dictionary storing the derivative operators.

It also creates a relative permeability, `mu_r`, as `numpy.ones(eps_r.shape)` and a source `src` as `numpy.zeros(eps_r.shape)`.

### Adding sources is exciting!

Sources can be added to the simulation either by manually editing the 2D src array inside of the simulation object,

	simulation.src[10,20:30] = 1

or by adding modal sources, which are defined as planes within the 2D domain which launch a mode in their normal direction. Modal source definitions can be added to the simulation by

	simulation.add_mode(neff, direction, center, width)
	simulation.setup_modes()

- `neff` : defines the effective index of the mode; this will be used as the eigenvalue guess
- `direction` : defines the normal direction of the plane, should be either 'x' or 'y'
- `center` : defines the center coordinates for the plane in cell coordinates [xc, yc]
- `width` : defines the width of the plane in number of cells

Note that `simulation.setup_modes()` must always be called after adding mode(s) in order to populate `simulation.src`.

### Solving for the electromagnetic fields

Now, we have everything we need to solve the system for the electromagnetic fields, by running

	fields = simulation.solve_fields(timing=False)

`simulation.src` is proportional to either the `Jz` or `Mz` source term, depending on whether `pol` is set to `'Ez'` or `'Hz'`, respectively.

`fields` is a tuple containing `(Ex, Ey, Hz)` or `(Hx, Hy, Ez)` depending on the polarization.

### Setting a new permittivity

If you want to change the permittivity distribution, reassigning `eps_r`

	simulation.eps_r = eps_new

will automatically solve for a new system matrix with the new permittivity distribution.  Note that `simulation.setup_modes()` should also be called if the permittivity changed within the plane of any of the modal sources. <- I'll make this happen automatically later -T

### Plotting

Primary fields (Hz/Ez) can be visualized using the included helper functions:

	simulation.plt_re(outline=True, cbar=True)
	simulation.plt_abs(outline=True, cbar=True)

These optionally outline the permittivity with contours and can be supplied with a matplotlib axis handle to plot into.

### Requirements

- numpy
- scipy
- matplotlib

To load the MKL solver:

	git submodule update --init --recursive

### To Do

#### Whenever
- [x] Modal source.
- [x] More dope plotting methods.
- [ ] xrange, yrange labels on plots.
- [ ] set modal source amplitude (and normalization)
- [ ] Add ability to run local jupyter notebooks running FDFD on parallel from hera.
- [ ] Save the factorization of `A` in the `Fdfd` object to be reused later if one has the same `A` but a different `b`.
- [ ] Allow the source term to have `(Jx, Jy, Jz, Mx, My, Mz)`, which would be useful for adjoint stuff where the source is not necessarily along the `z` direction.
- [x] Clean up imports (e.g. `import numpy as np` to `from numpy import abs, zeros, ...`)
