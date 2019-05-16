#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
Mechanical system.
"""

import numpy as np
from scipy.sparse import csc_matrix
from time import time

from ..mesh import Mesh
from ..assembly import Assembly
from ..boundary import DirichletBoundary
from ..observers import MaterialObserver, NodesObserver

__all__ = [
    'MechanicalSystem'
]


class MechanicalSystem:
    '''
    Master class for mechanical systems with the goal to black-box the routines of assembly and element selection.

    Attributes
    ----------
    mesh_class : instance of Mesh()
        Class handling the mesh.
    assembly_class : instance of Assembly()
        Class handling the assembly.
    dirichlet_class : instance of DirichletBoundary
        Class handling the Dirichlet boundary conditions.
    T_output : list of floats
        List of timesteps saved.
    u_output : list of ndarrays
        List of unconstrained displacement arrays corresponding to the timesteps in T_output.
    S_output : list of ndarrays
        List of stress arrays corresponding to the timesteps in T_output.
    E_output : list of ndarrays
        List of strain arrays corresponding to the timesteps in T_output.
    stress : ndarray
        Array of nodal stress of the last assembly run. Shape is (no_of_nodes, 6).
    strain : ndarray
        Array of nodal strain of the last assembly run. Shape is (no_of_nodes, 6).
    stress_recovery : bool
        Flag for option stress_recovery.
    iteration_info : ndarray
        Array containing the information of an iterative solution procedure. Eteration_info[:,0] is the time
        information, iteration_info[:,1] is the number of iterations, iteration_info[:,3] is the residual.
    M_constr : ?
        Mass matrix
    D_constr : ?
        Damping matrix
    '''

    def __init__(self, stress_recovery=False):
        '''
        Parameters
        ----------
        stress_recovery : bool, optional
            Flag, for setting stress recovery option. Default is False.
        '''

        self.stress_recovery = stress_recovery
        self.T_output = []
        self.u_output = []
        self.S_output = []
        self.E_output = []
        self.stress = None
        self.strain = None
        self.iteration_info = np.array([])

        # instantiate the important classes needed for the system
        self.mesh_class = Mesh()
        self.assembly_class = Assembly(self.mesh_class)
        self.dirichlet_class = DirichletBoundary()

        #  initialize observers
        self.material_observer = MaterialObserver(self)
        self.nodes_observer = NodesObserver(self)
        # make syntax a little bit leaner
        # !Christian Meyer: ! careful: This prohibits to easily change dirichlet_class instance, because old instance
        # still will be referenced!
        self.unconstrain_vec = self.dirichlet_class.unconstrain_vec
        self.constrain_vec = self.dirichlet_class.constrain_vec
        self.constrain_matrix = self.dirichlet_class.constrain_matrix

        # initializations to be overwritten by loading functions
        self.M_constr = None
        # TODO: Remove workaround for update of damping matrix self.D_constr >>>
        self.rayleigh_damping = False
        self.rayleigh_damping_alpha = None
        self.rayleigh_damping_beta = None
        self.D_constr = None
        # TODO: <<< Remove workaround for update of damping matrix self.D_constr
        self.no_of_dofs_per_node = None

        # external force to be overwritten by user-defined external forces
        # self._f_ext_unconstr = lambda t: np.zeros(self.mesh_class.no_of_dofs)

    def load_mesh_from_gmsh(self, msh_file, phys_group, material, scale_factor=1):
        '''
        Load the mesh from a msh-file generated by gmsh.

        Parameters
        ----------
        msh_file : str
            File name to an existing .msh file.
        phys_group : int
            Integer key of the physical group which is considered as the mesh part.
        material : amfe.Material
            Material associated with the physical group to be computed.
        scale_factor : float, optional
            Scale factor for the mesh to adjust the units. The default value is 1, i.e. no scaling is done.
        '''

        self.mesh_class.import_msh(msh_file, scale_factor=scale_factor)
        self.mesh_class.load_group_to_mesh(phys_group, material)

        # Add material observer
        material.add_observer(self.material_observer)

        self.no_of_dofs_per_node = self.mesh_class.no_of_dofs_per_node

        self.assembly_class.preallocate_csr()
        # Add Nodes Observer
        self.assembly_class.add_observer(self.nodes_observer)
        self.dirichlet_class.no_of_unconstrained_dofs = self.mesh_class.no_of_dofs
        self.dirichlet_class.update()

    def deflate_mesh(self):
        '''
        Remove free floating nodes not connected to a selected element from the mesh.
        '''

        self.mesh_class.deflate_mesh()
        self.assembly_class.preallocate_csr()
        self.dirichlet_class.no_of_unconstrained_dofs = self.mesh_class.no_of_dofs
        self.dirichlet_class.update()

    def load_mesh_from_csv(self, node_list_csv, element_list_csv, no_of_dofs_per_node=2, explicit_node_numbering=False,
                           ele_type=False):
        '''
        Loads the mesh from two csv-files containing the node and the element list.

        Parameters
        ----------
        node_list_csv: str
            Filename of the csv-file containing the coordinates of the nodes (x, y, z)
        element_list_csv: str
            Filename of the csv-file containing the nodes which belong to one element
        no_of_dofs_per_node: int, optional
            Degree of freedom per node as saved in the csv-file
        explicit_node_numbering : bool, optional
            Flag stating, if the node numbers are explcitly numbered in the csv file, i.e. if the first column gives
            the numbers of the nodes.
        ele_type: str
            Spezifiy elements type of the mesh (e.g. for a Tri-Mesh different elements types as Tri3, Tri4, Tri6 can be
            used). If not spezified value is set to 'False'.
        '''

        self.mesh_class.import_csv(node_list_csv, element_list_csv,
                                   explicit_node_numbering=explicit_node_numbering,
                                   ele_type=ele_type)
        self.no_of_dofs_per_node = no_of_dofs_per_node
        self.assembly_class.preallocate_csr()
        return

    def tie_mesh(self, master_key, slave_key, master_prop='phys_group', slave_prop='phys_group', tying_type='fixed',
                 verbose=False, conform_slave_mesh=False, fix_mesh_dist=1E-3):
        '''
        Tie nonconforming meshes for a given master and slave side.

        Parameters
        ----------
        master_key : int or string
            Mesh key of the master face mesh. The master face mesh has to be at least the size of the slave mesh. It is
            better, when the master mesh is larger than the slave mesh.
        slave_key : int or string
            Mesh key of the slave face mesh or point cloud
        master_prop : string, optional
            Mesh property for which master_key is specified. Default value: 'phys_group'.
        slave_prop : string, optional
            Mesh property for which slave_key is specified. Default value: 'phys_group'
        tying_type : string {'fixed', 'slide'}
            Mesh tying type. 'fixed' glues the meshes together while 'slide' allows for a sliding motion between the
            meshes.

        Notes
        -----
        The master mesh has to embrace the full slave mesh. If this is not the case, the routine will fail, a slave
        point outside the master mesh cannot be addressed to a specific element.
        '''

        vals = self.mesh_class.tie_mesh(master_key=master_key, slave_key=slave_key, master_prop=master_prop,
                                        slave_prop=slave_prop, tying_type=tying_type, verbose=verbose,
                                        fix_mesh_dist=fix_mesh_dist)

        self.dirichlet_class.add_constraints(*vals)
        self.dirichlet_class.update()
        return

    def apply_dirichlet_boundaries(self, key, coord, mesh_prop='phys_group'):
        '''
        Apply dirichlet-boundaries to the system.

        Parameters
        ----------
        key : int
            Key for mesh property which is to be chosen. Matches the group given in the gmsh file. For help, the
            function mesh_information or boundary_information gives the groups.
        coord : str {'x', 'y', 'z', 'xy', 'xz', 'yz', 'xyz'}
            Coordinates which should be fixed.
        mesh_prop : str {'phys_group', 'geom_entity', 'el_type'}, optional
            Label of which the element should be chosen from. Default is 'phys_group'.
        '''

        self.mesh_class.set_dirichlet_bc(key, coord, mesh_prop)
        self.dirichlet_class.constrain_dofs(self.mesh_class.dofs_dirichlet)
        return

    def apply_neumann_boundaries(self, key, val, direct, time_func=None, shadow_area=False, mesh_prop='phys_group'):
        '''
        Apply neumann boundaries to the system via skin elements.

        Parameters
        ----------
        key : int
            Key of the physical domain to be chosen for the neumann bc.
        val : float
            Value for the pressure/traction onto the element.
        direct : ndarray or str 'normal'
            Direction, in which force should act at.
        time_func : function object
            Function object returning a value between -1 and 1 given the input t:

            >>> val = time_func(t)

        shadow_area : bool, optional
            Flag, if force should be proportional to shadow area of surface with respect to direction. Default: False.
        mesh_prop : str {'phys_group', 'geom_entity', 'el_type'}, optional
            Label of which the element should be chosen from. Default is phys_group.
        '''

        self.mesh_class.set_neumann_bc(key=key, val=val, direct=direct, time_func=time_func, shadow_area=shadow_area,
                                       mesh_prop=mesh_prop)
        self.assembly_class.compute_element_indices()
        return

    def export_paraview(self, filename, field_list=None):
        '''
        Export the system with the given information to paraview.

        Parameters
        ----------
        filename : str
            Filename to which the xdmf file and the hdf5 file will be saved.
        field_list : list, optional
            List of tuples containing a field to be exported as well as a dictionary with the attribute information of
            the hdf5 file. Example:

                >>> # example field list with reduced displacement not to export
                >>> # ParaView and strain epsilon to be exported to ParaView
                >>> field_list = [(q_red, {'ParaView':False, 'Name':'q_red'}),
                                  (eps, {'ParaView':True,
                                         'Name':'epsilon',
                                         'AttributeType':'Tensor6',
                                         'Center':'Node',
                                         'NoOfComponents':6})]
        '''

        if field_list is None:
            field_list = []
        t1 = time()
        if len(self.T_output) is 0:
            self.T_output.append(0)
            self.u_output.append(np.zeros(self.mesh_class.no_of_dofs))
            if self.stress_recovery:
                self.S_output.append(np.zeros((self.mesh_class.no_of_nodes, 6)))
                self.E_output.append(np.zeros((self.mesh_class.no_of_nodes, 6)))
        print('Start exporting mesh for paraview to:\n    ', filename)

        if self.stress_recovery and len(self.S_output) > 0 \
                and len(self.E_output) > 0:
            no_of_timesteps = len(self.T_output)
            S_array = np.array(self.S_output).reshape((no_of_timesteps, -1))
            E_array = np.array(self.E_output).reshape((no_of_timesteps, -1))
            S_export = (S_array.T, {'ParaView': True,
                                    'Name': 'stress',
                                    'AttributeType': 'Tensor6',
                                    'Center': 'Node',
                                    'NoOfComponents': 6})
            E_export = (E_array.T, {'ParaView': True,
                                    'Name': 'strain',
                                    'AttributeType': 'Tensor6',
                                    'Center': 'Node',
                                    'NoOfComponents': 6})
            field_list.append(S_export)
            field_list.append(E_export)

        bmat = self.dirichlet_class.b_matrix()
        self.mesh_class.save_mesh_xdmf(filename, field_list, bmat, u=self.u_output, timesteps=self.T_output)
        t2 = time()
        print('Mesh for paraview successfully exported in ' +
              '{0:4.2f} seconds.'.format(t2 - t1))
        return

    def M(self, u=None, t=0, force_update=False):
        '''
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
        '''

        if self.M_constr is None or force_update:
            if u is not None:
                u_unconstr = self.unconstrain_vec(u)
            else:
                u_unconstr = None

            M_unconstr = self.assembly_class.assemble_m(u_unconstr, t)
            self.M_constr = self.constrain_matrix(M_unconstr)
        return self.M_constr

    def K(self, u=None, t=0):
        '''
        Compute and return the stiffness matrix of the mechanical system.

        Parameters
        ----------
        u : ndarray, optional
            Displacement field in voigt notation.
        t : float, optional
            Time.

        Returns
        -------
        K : sp.sparse.sparse_matrix
            Stiffness matrix with applied constraints in sparse CSC format.
        '''

        if u is None:
            u = np.zeros(self.dirichlet_class.no_of_constrained_dofs)

        K_unconstr = self.assembly_class.assemble_k_and_f(self.unconstrain_vec(u), t)[0]
        return self.constrain_matrix(K_unconstr)

    # TODO: Remove workaround for update of damping matrix self.D_constr >>>
    def D(self, u=None, t=0, force_update=False):
        '''
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
        '''

        if self.D_constr is None or force_update:
            if self.rayleigh_damping:
                self.D_constr = self.rayleigh_damping_alpha * self.M() + self.rayleigh_damping_beta * self.K()
            else:
                self.D_constr = csc_matrix(self.M().shape)
        return self.D_constr

    def apply_no_damping(self):
        '''
        Apply NO damping to the system, i.e. (re)set damping matrix to D = 0.
        '''

        self.rayleigh_damping = False
        self.rayleigh_damping_alpha = None
        self.rayleigh_damping_beta = None
        self.D(force_update=True)
        return

    def apply_rayleigh_damping(self, alpha, beta):
        '''
        Apply Rayleigh damping to the system, i.e. set damping matrix to D = alpha*M + beta*K(0). This is simple
        Rayleigh damping with regard to the system linearized around zero displacement.

        Parameters
        ----------
        alpha : float
            Damping coefficient w.r.t. the mass matrix.
        beta : float
            Damping coefficient w.r.t. the stiffness matrix.
        '''

        self.rayleigh_damping = True
        self.rayleigh_damping_alpha = alpha
        self.rayleigh_damping_beta = beta
        self.D(force_update=True)
        return

    # TODO: <<< Remove workaround for update of damping matrix self.D_constr

    def f_int(self, u, t=0):
        '''
        Return the elastic restoring force of the system.
        '''
        f_unconstr = self.assembly_class.assemble_k_and_f(self.unconstrain_vec(u), t)[1]
        return self.constrain_vec(f_unconstr)

    def _f_ext_unconstr(self, u, t):
        '''
        Return the unconstrained external force coming from the Neumann BCs. This function is just a placeholder if you
        want to change the behavior of f_ext: This function may be monkeypatched if necessary, for instance, when a
        global external force, e.g. gravity, should be applied.
        '''

        f_unconstr = self.assembly_class.assemble_k_and_f_neumann(self.unconstrain_vec(u), t)[1]
        return f_unconstr

    def f_ext(self, u=None, du=None, t=0):
        '''
        Return the nonlinear external force of the right hand side of the equation, i.e. the excitation.
        '''

        if u is None:
            u = np.zeros(self.dirichlet_class.no_of_constrained_dofs)
        return self.constrain_vec(self._f_ext_unconstr(u, t))

    def K_and_f(self, u=None, t=0):
        '''
        Compute tangential stiffness matrix and nonlinear force vector in one assembly run.
        '''

        if u is None:
            u = np.zeros(self.dirichlet_class.no_of_constrained_dofs)
        if self.stress_recovery:  # make sure, that current stress / strain is exported
            K_unconstr, f_unconstr, self.stress, self.strain = self.assembly_class.assemble_k_f_S_E(
                self.unconstrain_vec(u), t)
        else:
            K_unconstr, f_unconstr = self.assembly_class.assemble_k_and_f(self.unconstrain_vec(u), t)
        K = self.constrain_matrix(K_unconstr)
        f = self.constrain_vec(f_unconstr)
        return K, f
        return

    def write_timestep(self, t, u):
        '''
        Write the timestep to the mechanical_system class.
        '''

        self.T_output.append(t)
        self.u_output.append(self.unconstrain_vec(u))
        # Check both, if stress recovery and if stress and strain is there
        if self.stress_recovery:
            # catch the case when no stress was computed, for instance in time
            # integration
            if self.stress is None and self.strain is None:
                self.stress = np.zeros((self.mesh_class.no_of_nodes, 6))
                self.strain = np.zeros((self.mesh_class.no_of_nodes, 6))
            self.S_output.append(self.stress.copy())
            self.E_output.append(self.strain.copy())

    def clear_timesteps(self):
        '''
        Clear the timesteps gathered internally.
        '''

        self.T_output = []
        self.u_output = []
        self.S_output = []
        self.E_output = []
        self.stress = None
        self.strain = None
        self.iteration_info = np.array([])
        return

    def set_solver(self, solver, **solver_options):
        '''
        Set solver to be able to use shortcut my_system.solve() for solving the system.
        '''

        self.solver = solver(mechanical_system=self, **solver_options)
        return

    def solve(self):
        '''
        Shortcut to solve system.
        '''

        if not hasattr(self, 'solver'):
            raise ValueError('No solver set. Use my_system.set_solver(solver, **solver_options) to set solver first.')

        self.solver.solve()
        return