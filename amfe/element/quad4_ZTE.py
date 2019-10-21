#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
3d hexa8 element.
"""

__all__ = [
    'Quad4_ZTE'
]

import numpy as np

from .element import Element
from .tools import compute_B_matrix, scatter_matrix

# try to import Fortran routines
use_fortran = False
try:
    import amfe.f90_element
    use_fortran = True
except Exception:
    print('Python was not able to load the fast fortran element routines.')


class Quad4ZTE(Element):
    """
    Zero thickness element with two facing quads.

    .. code::


                 eta
        3----------2
        |\     ^   |\
        | \    |   | \
        |  \   |   |  \
        |   7------+---6
        |   |  +-- |-- | -> xi
        0---+---\--1   |
         \  |    \  \  |
          \ |     \  \ |
           \|   zeta  \|
            4----------5

    """
    name = 'Quad4_ZTE'

    def __init__(self, contact_models, *args, **kwargs, ):
        """
                Parameters
                ----------
                contact_models: amfe.Hysteresis_contact - object
                    Objects handling the contact, has to be equal to the number of Gauss Points
                """
        super().__init__(*args, **kwargs)

        self.contact_models = contact_models
        self.K = np.zeros((24, 24))
        self.f = np.zeros((24, 1))

        # Gauss-Point-Handling:
        g1 = np.sqrt(1/3)

        # Tupel for enumerator (xi, eta, weight)
        self.gauss_points = ((-g1, -g1, 1.),
                             (g1, -g1, 1.),
                             (g1, g1, 1.),
                             (-g1, g1, 1.))

        self.extrapolation_points = np.array([
            [1 + np.sqrt(3) / 2, -1 / 2, 1 - np.sqrt(3) / 2, -1 / 2],
            [-1 / 2, 1 + np.sqrt(3) / 2, -1 / 2, 1 - np.sqrt(3) / 2],
            [1 - np.sqrt(3) / 2, -1 / 2, 1 + np.sqrt(3) / 2, -1 / 2],
            [-1 / 2, 1 - np.sqrt(3) / 2, -1 / 2, 1 + np.sqrt(3) / 2]]).T

    @staticmethod
    def fields():
        return 'ux', 'uy', 'uz'

    def dofs(self):
        return (('N', 0, 'ux'),
                ('N', 0, 'uy'),
                ('N', 0, 'uz'),
                ('N', 1, 'ux'),
                ('N', 1, 'uy'),
                ('N', 1, 'uz'),
                ('N', 2, 'ux'),
                ('N', 2, 'uy'),
                ('N', 2, 'uz'),
                ('N', 3, 'ux'),
                ('N', 3, 'uy'),
                ('N', 3, 'uz'),  # end of top Quad
                ('N', 4, 'ux'),
                ('N', 4, 'uy'),
                ('N', 4, 'uz'),
                ('N', 5, 'ux'),
                ('N', 5, 'uy'),
                ('N', 5, 'uz'),
                ('N', 6, 'ux'),
                ('N', 6, 'uy'),
                ('N', 6, 'uz'),
                ('N', 7, 'ux'),
                ('N', 7, 'uy'),
                ('N', 7, 'uz'))  # end of bottom Quad

    def _compute_tensors(self, X, u, t):
        X_mat = X.reshape(8, 3)
        u_mat = u.reshape(8, 3)

        u_rel = u[0:4,:] - u[4:8,:]

        self.K *= 0
        self.f *= 0
        self.S *= 0
        self.E *= 0

        for n_gauss, (xi, eta, w) in enumerate(self.gauss_points):


            N = np.array([  [(-eta + 1)*(-xi + 1)/4],
                            [ (-eta + 1)*(xi + 1)/4],
                            [  (eta + 1)*(xi + 1)/4],
                            [ (eta + 1)*(-xi + 1)/4]])


            dN_dxi = np.array([ [ eta/4 - 1/4,  xi/4 - 1/4],
                                [-eta/4 + 1/4, -xi/4 - 1/4],
                                [ eta/4 + 1/4,  xi/4 + 1/4],
                                [-eta/4 - 1/4, -xi/4 + 1/4]])

            dx_dxi = X_mat[0:4,:].T @ dN_dxi
            # Compute the normal vector
            n = np.cross(dx_dxi[:, 1], dx_dxi[:, 0])

            # Vectors of local coordinates in global coordinates, should be A
            dX_dxi = np.zeros((3,3))
            dX_dxi[:,1] = dx_dxi[:, 1]
            dX_dxi[:, 2] = dx_dxi[:, 2]
            dX_dxi[:, 3] = n

            # Should be inverse of A
            dxi_dX = np.linalg.inv(dX_dxi)
            det = np.linalg.det(dX_dxi)

            # Should be dN_dX
            B0_tilde = dN_dxi @ dxi_dX
            # u_rel in local coordinates??
            u_rel_local = dxi_dX @ u_rel.T
            contact_model = self.contact_models[n]
            # This is the contact force and Jacobian in relative coordinates
            K_contact, f_contact = contact_model(u_rel_local) #I hope that this actually saves the state

            # Missing: Map it back

            self.K += K_contact * w
            self.f += f_contact * w

            # extrapolation of gauss element

        return




