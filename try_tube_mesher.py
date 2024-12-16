import sys

from numpy import linspace, pi, stack, cos, sin, arange
from pyvista import Plotter

from seagullmesh.tube_mesher import TubeMesher

# TODO: default behavior is flipped?
sm = TubeMesher.cylinder(n_radial=5, n_axial=8, closed=True, flip_faces=False, radius=0.5)
sm.vertex_data['idx'] = arange(sm.n_vertices)
assert sm.is_valid

points, faces = sm.to_polygon_soup()
centroids = points[faces].mean(axis=1)

normals = sm.faces.normals()
m = sm.to_pyvista(True)
p = Plotter(shape=(2, 2))
p.subplot(0, 0)
p.add_mesh(m.copy(), show_edges=True, scalars='t')

p.subplot(0, 1)
p.add_mesh(m.copy(), show_edges=True, scalars='theta')

p.subplot(1, 0)
p.add_mesh(m.copy(), show_edges=True, scalars='idx')

p.subplot(1, 1)
p.add_mesh(m.copy(), show_edges=True, scalars='is_cap')
p.add_arrows(m.cell_centers().points, normals * 0.1)

p.link_views()
p.show()
