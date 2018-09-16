import unittest
import numpy as np

from fdfdpy import Simulation

class Test_Simulation(unittest.TestCase):
    """ Tests the simulation object for various functionalities """

    def test_inputs(self):
        """ this is a quick example of ensuring that an error is thrown
        when passing certain arguments to Simulation """

        # the 'good' inputs
        Nx = 100
        Ny = 50
        omega = 100
        eps_r = np.ones((Nx, Ny))
        dl = 0.001
        NPML = [10, 10]
        pol = 'Hz'
        L0 = 1e-4

        # negative frequency
        with self.assertRaises(ValueError):
            Simulation(-omega, eps_r, dl, NPML, pol)

        # list of frequencies
        with self.assertRaises(ValueError):
            Simulation([100, 200, 300], eps_r, dl, NPML, pol)

        # negative epsilon of frequencies
        with self.assertRaises(ValueError):
            Simulation(omega, -eps_r, dl, NPML, pol)

        # list epsilon instead of numpy array
        with self.assertRaises(ValueError):
            Simulation(omega, list(eps_r), dl, NPML, pol)

        # negative dl
        with self.assertRaises(ValueError):
            Simulation(omega, eps_r, -dl, NPML, pol)

        # list of dl
        with self.assertRaises(ValueError):
            Simulation(omega, eps_r, [1e-4, 1e-5], NPML, pol)

        # NPML a number
        with self.assertRaises(ValueError):
            Simulation(omega, eps_r, dl, 10, pol)

        # NPML too many elements
        with self.assertRaises(ValueError):
            Simulation(omega, eps_r, dl, [10, 10, 10], pol)

        # NPML larger than domain
        with self.assertRaises(ValueError):
            Simulation(omega, eps_r, dl, [200, 200], pol)

        # polarization not a string
        with self.assertRaises(ValueError):
            Simulation(omega, eps_r, dl, NPML, 5)

        # polarization not the right string
        with self.assertRaises(ValueError):
            Simulation(omega, eps_r, dl, NPML, 'WrongPolarization')


if __name__ == '__main__':
    main()
