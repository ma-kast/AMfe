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
    'Quad4ZTE'
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

        # Here, we implicitly assume the ordering of the second Quad. It might be, that it actually is ordered counter-
        # clockwise. This is something, that might need to be fixed in a code. I.e. we need to take care of numbering.
        u_rel = u_mat[0:4, :] - u_mat[4:8, :]
        X_top = X_mat[0:4, :]
        self.K *= 0
        self.f *= 0

        for n_gauss, (xi, eta, w) in enumerate(self.gauss_points):

            N = np.array([[(-eta + 1) * (-xi + 1)/4],
                          [(-eta + 1) * (xi + 1)/4],
                          [(eta + 1) * (xi + 1)/4],
                          [(eta + 1) * (-xi + 1)/4]])

            dN_dxi = np.array([[eta/4 - 1/4,  xi/4 - 1/4],
                               [-eta/4 + 1/4, -xi/4 - 1/4],
                               [eta/4 + 1/4,  xi/4 + 1/4],
                               [-eta/4 - 1/4, -xi/4 + 1/4]])

            k_gp, f_gp = self._compute_gausspoint_contribution(N, dN_dxi, X_top, u_rel, n_gauss)
            # Scatter onto all nodal dofs
            f_mat_gp = np.outer(N, f_gp)
            k_mat_gp = self._assemble_big_k(k_gp, N)

            # Accumulate gauss point contribution for integration
            self.K[:12, :12] += k_mat_gp * w
            self.f[:12, :] += f_mat_gp.reshape(12, 1) * w

        # Scatter contributions to bottom element and interactions
        self.K[12:, 12:] = self.K[:12, :12]
        self.K[:12, 12:] = -self.K[:12, :12]
        self.K[12:, :12] = -self.K[:12, :12]
        self.f[12:, :] = -self.f[:12]

        return

    def _compute_gausspoint_contribution(self, N, dN_dxi, X_upper, u_rel, n_gauss):
        dx_dxi = X_upper.T @ dN_dxi
        # Compute the normal vector
        normal_vector = np.cross(dx_dxi[:, 1], dx_dxi[:, 0])
        normal_vector = normal_vector / np.linalg.norm(normal_vector, ord=2)

        # Vectors of local coordinates in global coordinates, such that g_local = dX_dxi g_global
        dX_dxi = np.zeros((3, 3))
        dX_dxi[:, 0] = dx_dxi[:, 0]
        dX_dxi[:, 1] = dx_dxi[:, 1]
        dX_dxi[:, 2] = -normal_vector

        # Find necessary inverse transform
        dxi_dX = np.linalg.inv(dX_dxi)
        det = np.linalg.det(dX_dxi)

        Minv = dxi_dX @ dxi_dX.T
        # Compute the relative displacement in local coordinates
        u_rel_local = dX_dxi.T @ u_rel.T @ N

        # Compute local contact state
        contact_model = self.contact_models[n_gauss]

        # We need to pass the metric Minv to correctly compute the norm of the trial stress
        K_contact_local, f_contact_local = contact_model.K_and_f(u_rel_local, Minv[:2, :2])

        # Transfer back to the original orientation. Inverse coordinate transformation and chain rule on u_local
        f_original = dxi_dX.T  @ f_contact_local * det
        K_original = dxi_dX.T  @ K_contact_local  @ dX_dxi.T * det
        return K_original, f_original

    def _assemble_big_k(self, k_mat, N):
        weights = np.outer(N, N)
        big_k = np.zeros((12, 12))
        for index_x in range(4):
            for index_y in range(4):
                big_k[index_x * 3:index_x * 3 + 3, index_y * 3:index_y * 3 + 3] = k_mat * weights[index_x, index_y]

        return big_k
