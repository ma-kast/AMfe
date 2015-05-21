#!/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 11:13:52 2015

Löschen aller Variablen in IPython:
%reset

Darstellung von Matrizen:
pylab.matshow(A)



@author: Johannes Rutzmoser
"""


import numpy as np
import scipy as sp
from scipy import sparse
from scipy import linalg

import multiprocessing as mp
from multiprocessing import Pool


class Assembly():
    '''
    Class for the more fancy assembly of a mesh and an element;
    In this state it is only possible to assemble homegeneous meshes;
    heterogeneous meshes should be assembled by another routine.
    '''
    def __init__(self, mesh, element):
        '''
        Parameters:
        ----

        mesh :      instance of the Mesh-class or a child of it

        element:    instance of the Element-class or a child of it

        Returns:
        ---
        None
        '''
        self.mesh = mesh
        self.element = element
        print('This Module is not implemented yet!\nDo not use!')
        pass

    def assemble_m(self, u):
        '''
        Assembles the mass matrix of the given mesh and element.

        Parameters:
        ---

        u :         nodal displacement of the nodes in Voigt-notation

        Returns:
        ---

        M :         unconstrained assembled mass matrix in sparse matrix coo
                    format.
        '''
        pass

    def assemble_k(self, u):
        '''
        Assembles the stiffness matrix of the given mesh and element.

        Parameters:
        ---

        u :         nodal displacement of the nodes in Voigt-notation

        Returns:
        ---

        K :         unconstrained assembled stiffness matrix in sparse matrix
                    coo format.
        '''
        pass

    def assemble_f(self, u):
        '''
        Assembles the force vector of the given mesh and element.

        Parameters:
        ---

        u :         nodal displacement of the nodes in Voigt-notation

        Returns:
        ---

        f :         unconstrained force vector in numpy.ndarray format of
                    dimension (ndim, )

        '''
        pass

    def assemble_k_and_f(self, u):
        '''
        Assembles the tangential stiffness matrix and the force matrix in one
        run as it is very often needed by an implicit integration scheme.

        Takes the advantage, that some element properties only have to be
        computed once.
        '''
        pass




class PrimitiveAssembly():
    '''
    Assemblierungsklasse, die für gegebene Tableaus von Knotenkoordinaten und Assemblierungsknoten eine Matrix assembliert
    '''

    # Hier muessen wir uns mal genau ueberlegen, was alles dem assembly uebergeben werden soll
    # ob das ganze Mesh, oder nur ein paar Attribute
    def __init__(self, nodes=None, elements=None, matrix_function=None, node_dof=2, vector_function=None):
        '''
        Verlangt ein dreispaltiges Koordinatenarray, indem die Koordinaten in x, y, und z-Koordinaten angegeben sind
        Anzahl der Freiheitsgrade für einen Knotenfreiheitsgrad: node_dof gibt an, welche Koordinaten verwendet werden sollen;
        Wenn mehr Koordinaten pro Knoten nötig sind (z.B. finite Rotationen), werden Nullen hinzugefügt
        '''
        self.nodes = nodes
        self.elements = elements
        self.matrix_function = matrix_function
        self.vector_function = vector_function
        self.node_dof = node_dof

        self.row_global = []
        self.col_global = []
        self.vals_global = []

        self.no_of_nodes = len(self.nodes)
        self.no_of_elements = len(self.elements)
        self.no_of_dofs = self.no_of_nodes*self.node_dof
        self.no_of_element_nodes = len(self.elements[0])

        self.ndof_global = self.no_of_dofs
        pass


    def assemble_matrix(self, u=None):
        '''
        assembliert die matrix_function für die Ursprungskonfiguration X und die Verschiebung u.
        '''
        # deletion of former variables
        self.row_global = []
        self.col_global = []
        self.vals_global = []
        # number of dofs per element (6 for triangle since no_of_element_nodes = 3 and node_dof = 2)
        ndof_local = self.no_of_element_nodes*self.node_dof
        # preset for u_local; necessary, when u=None
        u_local = np.zeros(ndof_local)

        for element in self.elements:
            # Koordinaten des elements
            X = np.array([self.nodes[i] for i in element]).reshape(-1)
            # element_indices have to be corrected in order respect the dimensions
            element_indices = np.array([[self.node_dof*i + j for j in range(self.node_dof)] for i in element]).reshape(-1)
            if u is not None:
                u_local = u[element_indices]
            element_matrix = self.matrix_function(X, u_local)
            row = np.zeros((ndof_local, ndof_local))
            row[:,:] = element_indices
            self.row_global.append(row.reshape(-1))
            self.col_global.append((row.T).reshape(-1))
            self.vals_global.append(element_matrix.reshape(-1))

        row_global_array = np.array(self.row_global).reshape(-1)
        col_global_array = np.array(self.col_global).reshape(-1)
        vals_global_array = np.array(self.vals_global).reshape(-1)
        Matrix_coo = sp.sparse.coo_matrix((vals_global_array, (row_global_array, col_global_array)), dtype=float)
        return Matrix_coo

    def assemble_vector(self, u):
        '''
        Assembliert die Force-Function für die Usprungskonfiguration X und die Verschiebung u
        '''
        global_force = np.zeros(self.ndof_global)
        for element in self.elements:
            X = np.array([self.nodes[i] for i in element]).reshape(-1)
            element_indices = np.array([[2*i + j for j in range(self.node_dof)] for i in element]).reshape(-1)
            global_force[element_indices] += self.vector_function(X, u[element_indices])
        return global_force







class MultiprocessAssembly():
    '''
    Klasse um schnell im Multiprozess zu assemblieren; Verteilt die Assemblierung auf alle Assemblierungsklassen und summiert die anschließend alles auf
    - funktioniert nicht so schnell, wie ich es erwartet hätte; genauere Analysen bisher noch nicht vorhanden, da profile-Tool nich zuverlässig für multiprocessing-Probleme zu funktionieren scheint.
    - ACHTUNG: Diese Klasse ist derzeit nicht in aktiver Nutzung. Möglicherweise macht es Sinn, diese Klasse zu überarbeiten, da sich die gesamte Programmstruktur gerade noch ändert.
    '''
    def __init__(self, assembly_class, list_of_matrix_functions, nodes_array, element_array):
        '''
        ???
        '''
        self.no_of_processes = len(list_of_matrix_functions)
        self.nodes_array = nodes_array
        self.element_array = element_array
        self.list_of_matrix_functions = list_of_matrix_functions
        domain_size = self.nodes_array.shape[0]//self.no_of_processes
        element_domain_list = []
        for i in range(self.no_of_processes - 1):
            element_domain_list.append(self.element_array[i*domain_size:(i+1)*domain_size,:])
        element_domain_list.append(self.element_array[(i+1)*domain_size:,:]) # assemble last domain to the end in order to consider flooring above
        self.assembly_class_list = [assembly_class(self.nodes_array, element_domain_list[i], matrix_function) for i, matrix_function in enumerate(list_of_matrix_functions)]
        pass

    def assemble(self):
        '''
        assembles the mesh with a multiprocessing routine
        '''
        pool = mp.Pool(processes=self.no_of_processes)
        results = [pool.apply_async(assembly_class.assemble) for assembly_class in self.assembly_class_list]
        matrix_coo_list = [j.get() for j in results]
        row_global = np.array([], dtype=int)
        col_global = np.array([], dtype=int)
        data_global = np.array([], dtype=float)
        for matrix_coo in matrix_coo_list:
            row_global = np.append(row_global, matrix_coo.row)
            col_global = np.append(col_global, matrix_coo.col)
            data_global = np.append(data_global, matrix_coo.data)
        matrix_coo = sp.sparse.coo_matrix((data_global, (row_global, col_global)), dtype=float)
        return matrix_coo



