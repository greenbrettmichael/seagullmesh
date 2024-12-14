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


corefined = sm0.corefiner(sm1).track(0, edge_constrained='edge_constrained').corefine()
n_components = sm0.label_connected_components(face_patches='face_patch')

c0 = sm0.to_pyvista(True)
c1 = sm1.to_pyvista(True)

p = Plotter()
p.add_mesh(c0, show_edges=True, scalars='face_patch')
# p.add_mesh(c1, show_edges=True)
p.show()
