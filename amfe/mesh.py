# Copyright (c) 2018, Lehrstuhl fuer Angewandte Mechanik, Technische Universitaet Muenchen.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#
"""
Mesh module of amfe.

This Module provides a mesh class that handles the mesh information:
Nodes, Mesh-Topology, Elementshapes, Groups, Ids
"""


import numpy as np
import _pickle as pickle

__all__ = [
    'Mesh',
]

# Describe Element shapes, that can be used in Amfe
# 2D Volume Elements:
element_2d_set = {'Tri6', 'Tri3', 'Tri10', 'Quad4', 'Quad8', }
# 3D Volume Elements:
element_3d_set = {'Tet4', 'Tet10', 'Hexa8', 'Hexa20', 'Prism6'}
# 2D Boundary Elements
boundary_2d_set = {'straight_line', 'quadratic_line'}
# 3D Boundary Elements
boundary_3d_set = {'straight_line', 'quadratic_line',
                   'Tri6', 'Tri3', 'Tri10', 'Quad4', 'Quad8'}


class Mesh:
    """
    Class for handling the mesh operations.

    Attributes
    ----------
    nodes : ndarray
        Array of x-y-z coordinates of the nodes in reference configuration. Dimension is
        (no_of_nodes, 2) for 2D problems and (no_of_nodes, 3) for 3D problems.
        z-direction is dropped for 2D problems!
    nodeid2idx : dict
        Dictionary with key = node-id: value = row id in self.nodes array for getting nodes coordinates X
    connectivity : list
        List of node-rowindices of self.nodes belonging to one element.
    ele_shapes : list
        List of element shapes. The list contains strings that describe the shape of the elements
    boundary_connectivity : list
        list of element connectivities on the boundary (node-rowindices of self.nodes)
    boundary_ele_shapes : list
        List of element shapes of the boundary mesh. The list contains strings that describe the shape of
        the boundary elements
    eleid2idx : dict
        Dictionary with key = element-id: value = (0/1, idx) tuple with
        0 = connectivity list, 1 = boundary_connectivity list, idx = idx of element in this list
    groups : list
        List of groups containing ids (not row indices!)
    """
    def __init__(self, dimension=3):
        """
        Parameters
        ----------
        dimension : int
            describes the dimension of the mesh (2 or 3)

        Returns
        -------
        None
        """
        # -- GENERAL INFORMATION --
        self._dimension = dimension

        # -- NODE INFORMATION --
        # node coordinates as np.array
        self.nodes = np.empty((0, dimension), dtype=float)
        # map from nodeid to idx in nodes array
        self.nodeid2idx = dict([])

        # -- ELEMENT INFORMATION --
        # connectivity for volume elements and list of shape information of each element
        # list of elements containing rowidx of nodes array of connected nodes in each element
        self.connectivity = list()
        self.ele_shapes = list()

        # the same for boundary elements
        self.boundary_connectivity = list()
        self.boundary_ele_shapes = list()

        # map from elementid to idx in connectivity and boundary connectivity lists
        # { id : (0/1, idx) } with:
        #   id = id of element
        #   0 = internal element , 1 = boundary element
        #   idx in connectivity or boundary_connectivity list respectively
        self.eleid2idx = dict([])

        # group dict with names mapping to element ids or node ids, respectively
        self.groups = dict()

    @property
    def no_of_nodes(self):
        """
        Returns the number of nodes

        Returns
        -------
        no_of_nodes: int
            Number of nodes of the whole mesh.
        """
        return self.nodes.shape[0]

    @property
    def no_of_elements(self):
        """
        Returns the number of volume elements

        Returns
        -------
        no_of_elements : int
            Number of volume elements in the mesh
        """
        return len(self.connectivity)

    @property
    def no_of_boundary_elements(self):
        """
        Returns the number of boundary elements

        Returns
        -------
        no_of_elements : int
            Number of boundary elements in the mesh
        """
        return len(self.boundary_connectivity)

    @property
    def dimension(self):
        """
        Returns the dimension of the mesh

        Returns
        -------
        dimension : int
            Dimension of the mesh
        """
        return self._dimension

    @property
    def nodes_voigt(self):
        """
        Returns the nodes in voigt notation

        Returns
        -------
        nodes_voigt : ndarray
            Returns the nodes in voigt-notation
        """
        return self.nodes.reshape(-1)

    def save(self, filename):
        """
        Shortcut for saving a mesh in a pickle file.

        Warning! This shortcut is not intended for long term savings for meshes.
        Use an exporter from IO Module instead.

        Parameters
        ----------
        filename : str
            Filename where the mesh should be saved

        Returns
        -------
            None
        """
        with open(filename, 'wb') as infile:
            pickle.dump(self, infile)

    # -- GETTER CLASSES NAMING CONVENTION --
    # The following classes are 'getter' classes.
    # We use the following naming convention for function names:
    #   get_<node|element><id|idx>[s]_by_<group|id|idx>[s]
    #            |            |     |     |        |      - Plural if signature accepts list of entities
    #            |            |     |     |        - Describe which entity is passed group, id or row index
    #            |            |     |     - 'by' keyword
    #            |            |      - plural s if list or single entity is returned
    #            |            - describes weather ids or row indices are returned
    #             - describes weather nodes or elements are returned
    # ---------------------------------------

    def get_elementidxs_by_group(self, group):
        """
        Returns elementindices of the ele_shape/boundary_shape property belonging to group

        Parameters
        ----------
        group : string
            groupname

        Returns
        -------
        elementidxes : list
            returns list of tuples (0/1, idx), where 0 = volume element, 1 = boundary element
        """
        elementids = self.groups[group]['elements']
        return [self.eleid2idx[elementid] for elementid in elementids]

    def get_elementidxs_by_groups(self, groups):
        """
        Returns elementindices of the ele_shape/boundary_shape property belonging to groups

        Parameters
        ----------
        groups : list
            groupnames as strings in a list

        Returns
        -------
            list of tuples (0/1, idx), where 0 = volume element, 1 = boundary element
        """
        elementids = list()
        for group in groups:
            elementids.extend(self.groups[group]['elements'])
        return [self.eleid2idx[elementid] for elementid in elementids]

    def get_nodeidxs_by_group(self, group):
        """
        Returns nodeindices of the nodes property belonging to a group

        Parameters
        ----------
        group : string or -1
            If a string is passed, the parameter is considered as groupname
            If -1 is passed it returns all nodes

        Returns
        -------
            nodeidxs : ndarray
        list of row indices of nodes selected by group

        """
        nodeids = []
        elementids = []
        if self.groups[group]['elements'] is not None:
            elementids = self.groups[group]['elements']
        if self.groups[group]['nodes'] is not None:
            nodeids = self.groups[group]['nodes']
        nodeidxs = [self.nodeid2idx[nodeid] for nodeid in nodeids]
        eledict = {0: self.connectivity, 1: self.boundary_connectivity}
        nodes = [eledict[ele_tuple[0]][ele_tuple[1]] for ele_tuple in
                 [self.eleid2idx[elementid] for elementid in elementids]]
        uniquenodes = set(nodeidxs)
        for ar in nodes:
            uniquenodes = uniquenodes.union(set(ar))
        return np.array(list(uniquenodes), dtype=np.int)

    def get_nodeidxs_by_groups(self, groups):
        """
        Returns nodeindieces of the nodes property belonging to a group

        Parameters
        ----------
        groups : list
            contains the groupnames as strings

        Returns
        -------
        nodeidxs : ndarray

        """
        nodeids = []
        elementids = []
        for group in groups:
            if group == -1:
                return [i for i in range(0, self.no_of_nodes)]
            if self.groups[group]['elements'] is not None:
                elementids.extend(self.groups[group]['elements'])
            if self.groups[group]['nodes'] is not None:
                nodeids.extend(self.groups[group]['nodes'])
        nodeidxs = [self.nodeid2idx[nodeid] for nodeid in nodeids]
        eledict = {0: self.connectivity, 1: self.boundary_connectivity}
        nodes = [eledict[ele_tuple[0]][ele_tuple[1]] for ele_tuple in
                 [self.eleid2idx[elementid] for elementid in elementids]]
        uniquenodes = set(nodeidxs)
        for ar in nodes:
            uniquenodes = uniquenodes.union(set(ar))
        return np.array(list(uniquenodes), dtype=np.int)

    def get_ele_shapes_by_idxs(self, elementidxes):
        """
        Returns list of element_shapes for elementidxes

        Parameters
        ----------
        elementidxes : list
            list of tuples (0/1, idx), where 0 = volume element, 1 = boundary_element

        Returns
        -------
        ele_shapes : list
            list of element_shapes as string
        """
        shapedict = {0: self.ele_shapes, 1: self.boundary_ele_shapes}
        return [shapedict[eleidx[0]][eleidx[1]] for eleidx in elementidxes]

    def get_nodeidxs_by_all(self):
        """
        Returns all nodeidxs

        Returns
        -------
        nodeidxs : list
            returns all nodeidxs
        """
        return np.array([i for i in range(0, self.no_of_nodes)], dtype=np.int)
