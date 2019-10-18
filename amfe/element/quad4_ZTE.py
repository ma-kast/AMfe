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


class Quad4_ZTE(Element):
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

    def __init__(self, contact_model, *args, **kwargs, ):
        """
                Parameters
                ----------
                contact_model: amfe.Hysteresis_contact - object
                    Object handling the contact
                """
        super().__init__(*args, **kwargs)

        self.contact_model = contact_model
        self.K = np.zeros((24,24))
        self.f = np.zeros(24)

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
        return ('ux', 'uy', 'uz')

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
                ('N', 3, 'uz'), # end of top Quad
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
                ('N', 7, 'uz')) # end of bottom Quad

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

            K_contact, f_contact = self.c
            dX_dxi = X_mat.T @ dN_dxi
            dxi_dX = np.linalg.inv(dX_dxi)
            det = np.linalg.det(dX_dxi)
            B0_tilde = dN_dxi @ dxi_dX
            H = u_mat.T @ B0_tilde
            F = H + np.eye(3)
            E = 1/2*(H + H.T + H.T @ H)
            S, S_v, C_SE = self.material.S_Sv_and_C(E)
            B0 = compute_B_matrix(B0_tilde, F)
            K_geo_small = B0_tilde @ S @ B0_tilde.T * det
            K_geo = scatter_matrix(K_geo_small, 3)
            K_mat = B0.T @ C_SE @ B0 * det

            self.K += (K_geo + K_mat) * w
            self.f += B0.T @ S_v * det * w

            # extrapolation of gauss element
            extrapol = self.extrapolation_points[:,n_gauss:n_gauss+1]
            self.S += extrapol @ np.array([[S[0,0], S[0,1], S[0,2],
                                            S[1,1], S[1,2], S[2,2]]])
            self.E += extrapol @ np.array([[E[0,0], E[0,1], E[0,2],
                                            E[1,1], E[1,2], E[2,2]]])
        return




