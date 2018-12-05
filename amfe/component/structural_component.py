# Copyright (c) 2018, Lehrstuhl für Angewandte Mechanik, Technische Universität München.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#

import numpy as np
from scipy.sparse import csc_matrix

from .mesh_component import MeshComponent
from amfe.constraint.structural_constraint_manager import  StructuralConstraintManager
from amfe.assembly.structural_assembly import StructuralAssembly
from amfe.component.constants import ELEPROTOTYPEHELPERLIST
from amfe.mesh import Mesh


class StructuralComponent(MeshComponent):
        
    TYPE = 'StructuralComponent'
    ELEMENTPROTOTYPES = dict(((element[0], element[1]()) for element in ELEPROTOTYPEHELPERLIST
                              if element[1] is not None))
    BOUNDARYELEMENTFACTORY = dict(((element[0], element[2]) for element in ELEPROTOTYPEHELPERLIST
                                   if element[2] is not None))
    VALID_GET_MAT_NAMES = ('K', 'M', 'D')

    def __init__(self, mesh=Mesh()):
        super().__init__(mesh)
        self.rayleigh_damping = None
        if mesh.dimension == 3:
            self._fields = ('ux', 'uy', 'uz')
        elif mesh.dimension == 2:
            self._fields = ('ux', 'uy')
        self._constraints = StructuralConstraintManager()
        self._assembly = StructuralAssembly()
        self._M_constr = None
        self._D_constr = None
        self._C_csr = None
        self._M_csr = None
        self._f_glob = None

    def M(self, u=None, t=0, force_update=False):
        """
        Compute and return the mass matrix of the mechanical system.

        Parameters
        ----------
        u : ndarray, optional
            Array of the displacement.
        t : float
            Time.
        force_update : bool
            Flag to force update of M otherwise already calculated M is returned. Default is False.

        Returns
        -------
        M : sp.sparse.sparse_matrix
            Mass matrix with applied constraints in sparse CSC format.
        """

        if self._M_constr is None or force_update:
            if u is not None:
                u_unconstr = self._constraints.unconstrain_u(u, t)
            else:
                u_unconstr = None

            self._M_csr.data[:] = 0.0
            self._assembly.assemble_m(self._M_csr, self._mesh.nodes_df, self.ele_obj,
                                      self._ele_obj_df.join(self._mesh.el_df)['connectivity'].values,
                                      self._mapping.elements2global, u_unconstr, t)
            self._M_constr = self._constraints.constrain_m(self._M_csr)
        return self._M_constr

    def D(self, u=None, t=0, force_update=False):
        """
        Compute and return the damping matrix of the mechanical system. At the moment either no damping
        (rayleigh_damping = False) or simple Rayleigh damping applied to the system linearized around zero
        displacement (rayleigh_damping = True) are possible. They are set via the functions apply_no_damping() and
        apply_rayleigh_damping(alpha, beta).

        Parameters
        ----------
        u : ndarray, optional
            Displacement field in voigt notation.
        t : float, optional
            Time.
        force_update : bool
            Flag to force update of D otherwise already calculated D is returned. Default is False.

        Returns
        -------
        D : scipy.sparse.sparse_matrix
            Damping matrix with applied constraints in sparse CSC format.
        """

        if self._D_constr is None or force_update:
            if self.rayleigh_damping:
                self._D_constr = self.rayleigh_damping[0] * self.M() + self.rayleigh_damping[1] * self.K()
            else:
                self._D_constr = csc_matrix(self.M().shape)
        return self._D_constr

    def f_int(self, u=None, t=0):
        """
        Compute and return the nonlinear internal force vector of the structural component.

        Parameters
        ----------
        u : ndarray, optional
            Displacement field in voigt notation. len(u) is equal to the number of dofs after constraints have been
            applied
        t : float, optional
            Time.

        Returns
        -------
        f_int : ndarray
            Nonlinear internal force vector after constraints have been applied
        """

        if u is None:
            u = np.zeros(self._constraints.no_of_constrained_dofs)

        f_unconstr = self._assembly.assemble_k_and_f(self._mesh.nodes_df, self.ele_obj,
                                                     self._ele_obj_df.join(self._mesh.el_df)['connectivity'].values,
                                                     self._mapping.elements2global,
                                                     self._constraints.unconstrain_u(u, t), t,
                                                     self._C_csr, self._f_glob)[1]
        return self._constraints.constrain_f_int(f_unconstr)

    def K(self, u=None, t=0):
        """
        Compute and return the stiffness matrix of the structural component

        Parameters
        ----------
        u : ndarray, optional
            Displacement field in voigt notation. len(u) is equal to the number of dofs after constraints have been
            applied
        t : float, optional
            Time.

        Returns
        -------
        K : sp.sparse.sparse_matrix
            Stiffness matrix with applied constraints in sparse CSC format.
        """

        if u is None:
            u = np.zeros(self._constraints.no_of_constrained_dofs)

        self._assembly.assemble_k_and_f(self._mesh.nodes_df, self.ele_obj,
                                        self._ele_obj_df.join(self._mesh.el_df)['connectivity'].values,
                                        self._mapping.elements2global, self._constraints.unconstrain_u(u, t), t,
                                        self._C_csr, self._f_glob)
        return self._constraints.constrain_k(self._C_csr)

    def K_and_f_int(self, u=None, t=0):
        """
        Compute and return the tangential stiffness matrix and internal force vector of the structural component.

        Parameters
        ----------
        u : ndarray, optional
            Displacement field in voigt notation. len(u) is equal to the number of dofs after constraints have been
            applied
        t : float, optional
            Time.

        Returns
        -------
        K : sp.sparse.sparse_matrix
            Stiffness matrix with applied constraints in sparse CSC format.
        f : ndarray
            Internal nonlinear force vector after constraints have been applied
        """

        if u is None:
            u = np.zeros(self._constraints.no_of_constrained_dofs)

        self._assembly.assemble_k_and_f(self._mesh.nodes_df, self.ele_obj,
                                        self._ele_obj_df.join(self._mesh.el_df)['connectivity'].values,
                                        self._mapping.elements2global, self._constraints.unconstrain_u(u, t), t,
                                        self._C_csr, self._f_glob)
        return self._constraints.constrain_k(self._C_csr), self._constraints.constrain_f_int(self._f_glob)
