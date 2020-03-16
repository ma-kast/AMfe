import meshio


filename = '/Users/makast/PycharmProjects/AMfe/experiments/meshes_trelis/Exports_coarse_mesh/BRB_1.e'
mesh = meshio.read(
    filename,
)

outputpath = '/Users/makast/PycharmProjects/AMfe/experiments/meshes_trelis/Exports_coarse_mesh/BRB_2.msh'
out = meshio.write(outputpath, mesh, file_format='gmsh')