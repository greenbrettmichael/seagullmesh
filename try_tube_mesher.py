import sys

from numpy import linspace, pi, stack, cos, sin
from pyvista import Plotter

from seagullmesh.tube_mesher import TubeMesher
sm = TubeMesher.cylinder(n_radial=4, n_axial=10, closed=True)

m0 = sm.to_pyvista(vertex_data=True, face_data=True)

print(sm.face_data['is_cap'][sm.faces])
sm.remove_connected_face_patches(to_remove=[True], face_patches=sm.face_data['is_cap'])
print(sm.face_data['is_cap'][sm.faces])
sys.exit()
m1 = sm.to_pyvista(vertex_data=True, face_data=True)

p = Plotter(shape=(1, 2))
p.subplot(0, 0)
p.add_mesh(m0, scalars='is_cap', show_edges=True)
p.subplot(0, 1)
p.add_mesh(m1, scalars='is_cap', show_edges=True)
p.show()