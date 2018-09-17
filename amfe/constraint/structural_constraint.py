#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
Super class of all structural constraint algorithms.
"""

__all__ = [
    'StructuralConstraint'
]


class StructuralConstraint:
    """
    Super class of all structural constraints.
    """

    def __init__(self, *args, **kwargs):
        pass

    def c(self, X_local, u_local, du_local, ddu_local, t):
        raise NotImplementedError('The c constraint is not implemented')

    def b(self, X_local, u_local, du_local, ddu_local, t):
        raise NotImplementedError('The c constraint is not implemented')

    def u_slave(self, X_local, u_local, du_local, ddu_local, t):
        raise NotImplementedError('The c constraint is not implemented')

    def du_slave(self, X_local, u_local, du_local, ddu_local, t):
        raise NotImplementedError('The c constraint is not implemented')

    def ddu_slave(self, X_local, u_local, du_local, ddu_local, t):
        raise NotImplementedError('The c constraint is not implemented')
