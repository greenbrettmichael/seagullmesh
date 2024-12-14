import sys

from numpy import linspace, pi, stack, cos, sin, arange
from pyvista import Plotter

from seagullmesh.tube_mesher import TubeMesher
sm = TubeMesher.cylinder(n_radial=4, n_axial=10, closed=True)
# sm.fd['is_cap1'] = sm.fd['is_cap'][:].astype(int)
ic1 = sm.fd.create('is_cap1', default=-10)
ic1[sm.fs] = sm.fd['is_cap'][:].astype(int).copy()

assert set(sm.fd['is_cap1']) == {0, 1}
sm.fd['idx'] = arange(sm.nf)

m0 = sm.to_pyvista(True)

# sm.remove_connected_face_patches(to_remove=[1], face_patches='is_cap1')
sm.remove_connected_face_patches(to_remove=list(range(5, 10)), face_patches='idx')

m1 = sm.to_pyvista(True)

print(m0.cell_data['idx'])
print(m1.cell_data['idx'])

p = Plotter(shape=(1, 2))
p.subplot(0, 0)
p.add_mesh(m0, scalars='idx', show_edges=True)
p.subplot(0, 1)
p.add_mesh(m1, scalars='idx', show_edges=True)
p.link_views()
p.show()