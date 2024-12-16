import sys

from numpy import linspace, pi, stack, cos, sin, arange
from pyvista import Plotter, PolyData

from seagullmesh.tube_mesher import TubeMesher
import pyvista as pv

# TODO: default behavior is flipped?
sm = TubeMesher.cylinder(n_radial=5, n_axial=8, closed=True, flip_faces=True, radius=0.5, triangulate=True)
sm.vertex_data['idx'] = arange(sm.n_vertices)
assert sm.is_valid

points, faces = sm.to_polygon_soup()
# centroids = points[faces].mean(axis=1)
centroids = [points[f].mean() for f in faces]

normals = sm.faces.normals()
# m = sm.to_pyvista(True)
m = PolyData()
m.points = points
m.faces = pv.CellArray.from_irregular_cells(faces)
m.point_data['t'] = sm.vertex_data['t'][:]
m.point_data['theta'] = sm.vertex_data['theta'][:]
m.point_data['idx'] = sm.vertex_data['idx'][:]
m.cell_data['is_cap'] = sm.face_data['is_cap'][:]

p = Plotter(shape=(2, 2))
p.subplot(0, 0)
p.add_mesh(m.copy(), show_edges=True, scalars='t')

p.subplot(0, 1)
p.add_mesh(m.copy(), show_edges=True, scalars='theta')

p.subplot(1, 0)
p.add_mesh(m.copy(), show_edges=True, scalars='idx')

p.subplot(1, 1)
p.add_mesh(m.copy(), show_edges=True, scalars='is_cap', opacity=0.5)
p.add_arrows(m.cell_centers().points, normals * 0.1)

p.link_views()
p.show()
