#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#
"""
Static Pipe example
"""

import numpy as np
import matplotlib.pyplot as plt
from amfe.io import amfe_dir
from amfe.ui import *
from amfe.material import KirchhoffMaterial
from amfe.solver import AmfeSolution, SolverFactory
from amfe.solver.translators import create_constrained_mechanical_system_from_component

input_file = amfe_dir('experiments/minimal_example_separate.msh')
output_file = amfe_dir('experiments/test')

mesh = import_mesh_from_file(input_file)

print(mesh)
print(mesh.dimension)
print(mesh.no_of_elements)
print(mesh.connectivity)
print(mesh.el_df)
print(mesh.no_of_boundary_elements)
elemend_ids = range(1,8388)
print(mesh.get_connectivity_by_elementids([1]))

my_component = create_structural_component(mesh)

my_material = KirchhoffMaterial(E=200E6, nu=0.3, rho=1E3)


assign_material_by_group(my_component, my_material, 0)

#set_dirichlet_by_nodeids(my_component, [39, 17, 19, 60,  3, 10, 16,  4], ('ux', 'uy', 'uz'))
my_system, formulation = create_constrained_mechanical_system_from_component(my_component, constant_mass=True, constant_damping=True,
                                                             constraint_formulation='boolean')


print(my_component.mapping.no_of_dofs)
print(my_system.dimension)
x0 = np.zeros(my_component.mapping.no_of_dofs)
K= my_component.K(x0,x0,0)
print(K)
plt.figure()
plt.spy(K)
plt.show()
print(my_component.get_physics())

