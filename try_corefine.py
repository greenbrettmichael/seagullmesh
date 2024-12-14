import numpy as np
from numpy import array, zeros
from pyvista import Plotter

from seagullmesh import Mesh3, Point3
from seagullmesh.tube_mesher import TubeMesher

ni = nj = 3
w = h = 5
z = 0


def calculator(i: int, j: int):
    x = w * (i / ni - 0.5)
    y = h * (j / nj - 0.5)
    return Point3(x, y, z)


sm0 = Mesh3.grid(ni=ni, nj=nj, calculator=calculator).triangulate_faces()
sm1 = TubeMesher.cylinder(n_radial=4, n_axial=2, closed=False)

m0 = sm0.to_pyvista(True)
m1 = sm1.to_pyvista(True)

corefined = sm0.corefiner(sm1).track(0, edge_is_constrained='edge_is_constrained').corefine()

face_patches = sm0.face_data.create('face_patch', default=0)
n_components = sm0.label_connected_components(
    face_patches=face_patches,
    edge_is_constrained='edge_is_constrained'
)

tree = sm0.aabb_tree()
face, _ = tree.locate_point(np.zeros(3))
cmp_idx = face_patches[face]
sm0.remove_connected_face_patches(to_remove=[cmp_idx], face_patches=face_patches)

sm0.to_pyvista(True).plot(show_edges=True)
#
# c0 = sm0.to_pyvista(True)
# c1 = sm1.to_pyvista(True)
#
# p = Plotter()
# p.add_mesh(c0, show_edges=True, scalars='face_patch')
# # p.add_mesh(c1, show_edges=True)
# p.show()
