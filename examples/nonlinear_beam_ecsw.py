# Beam example

# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#
"""
Example showing a cantilever beam which is loaded on the tip with a force
showing nonlinear displacements.
"""

import numpy as np

import amfe



input_file = amfe.amfe_dir('meshes/gmsh/bar.msh')
output_file = amfe.amfe_dir('results/beam_nonlinear/beam_ecsw')


my_material = amfe.KirchhoffMaterial(E=210E9, nu=0.3, rho=1E4, plane_stress=True)
my_system = amfe.MechanicalSystem()
my_system.load_mesh_from_gmsh(input_file, 7, my_material)
my_system.apply_dirichlet_boundaries(8, 'xy') # fixature of the left side
my_system.apply_neumann_boundaries(key=9, val=1E8, direct=(0,-1),
                                   time_func=lambda t: np.sin(31*t))

solverlin = amfe.LinearStaticsSolver(my_system)
solverlin.solve()
my_system.export_paraview(output_file + '_linear')


solvernl = amfe.NonlinearStaticsSolver(my_system, number_of_load_steps=50)
solvernl.solve()
my_system.export_paraview(output_file + '_nonlinear')

solverti = amfe.GeneralizedAlphaNonlinearDynamicsSolver(my_system, dt=0.001, t_end=1)
my_system.clear_timesteps()
solverti.solve()
my_system.export_paraview(output_file + '_nonlinear_ti')

# Basis generation:

omega, V = amfe.reduced_basis.vibration_modes(my_system,6)
Theta = amfe.reduced_basis.modal_derivatives(V,omega,my_system.K,my_system.M())

V_extended = amfe.augment_with_derivatives(V, Theta)

# Training Set Generation
nskts = amfe.hyper_red.compute_nskts(my_system)

my_system.clear_timesteps()
for i in range(nskts.shape[1]):
    my_system.write_timestep(i,nskts[:,i])

my_system.export_paraview(output_file + '_nskts')


# Reduce system
my_red_system = amfe.reduce_mechanical_system(my_system, V_extended)
ndofs = V_extended.shape[1]
initial_conditions = {'q0': np.zeros(ndofs), 'dq0': np.zeros(ndofs)}
solverti_red = amfe.GeneralizedAlphaNonlinearDynamicsSolver(my_red_system, dt=0.001, t_end=1, initial_conditions=initial_conditions)
my_red_system.clear_timesteps()

solverti_red.solve()
my_red_system.export_paraview(output_file + '_nonlinear_ti_red')


# Hyperreduction

my_hyperred = amfe.reduce_mechanical_system_ecsw(my_system,V_extended)
q_training = np.linalg.solve((V_extended.T @ V_extended), V_extended.T @ nskts)
my_hyperred.reduce_mesh(q_training)

my_hyperred.export_paraview(output_file + '_weights_ecsw')
solverti_hyperred = amfe.GeneralizedAlphaNonlinearDynamicsSolver(my_hyperred, dt=0.001, t_end=1, initial_conditions=initial_conditions)
my_hyperred.clear_timesteps()
solverti_hyperred.solve()
my_hyperred.export_paraview(output_file + '_nonlinear_ti_hyperred')

print('END')

# OLD Version
# from .. experiments.benchmarks.cantilever import my_system, dt, T, \
# input_file, output_file
#
# amfe.solve_nonlinear_displacement(my_system,300)
#
# my_system.export_paraview(output_file + '_full')
#
# #-----------------------------------------------------------------
# # Basis generation
# #-----------------------------------------------------------------
#
# # Eigenmodes
# [omega, Phi] = amfe.vibration_modes(my_system,5)
#
# sp.save(output_file + '_eigenfrequencies',omega)
# sp.save(output_file + '_eigenmodes',Phi)
#
# # SMDs
# Theta = amfe.static_correction_theta(Phi,my_system.K)
#
# sp.save(output_file + '_smds',Theta)
#
# V_red = amfe.linear_qm_basis(Phi,Theta)
#
# #-----------------------------------------------------------------
# # Build reduced system
# #-----------------------------------------------------------------
# my_system_red = amfe.reduce_mechanical_system(my_system,V_red)
#
# amfe.solve_nonlinear_displacement(my_system_red,300)
#
# my_system_red.export_paraview(output_file + '_red')
#
#
# #-----------------------------------------------------------------
# # Generate Krylov Force subspace
# #-----------------------------------------------------------------
#
# NSKTS = amfe.compute_nskts(my_system_red,None,4,8,3,20)
#
# my_system_nskts = copy.deepcopy(my_system_red)
#
# my_system_nskts.clear_timesteps()
#
# for i in range(NSKTS.shape[1]):
#     my_system_nskts.write_timestep(i,NSKTS[:,i])
#
# my_system_nskts.export_paraview(output_file + '_nskts')
#
# #------------------------------------------------------------------
# # Hyperreduce system
# #------------------------------------------------------------------
#
# my_system_hyperred = amfe.HyperRedSystem()