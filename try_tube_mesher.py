
from numpy import linspace, pi, stack, cos, sin, arange
from pyvista import Plotter, PolyData
from seagullmesh import Point3
from seagullmesh.tube_mesher import TubeMesher
import pyvista as pv
import numpy as np
n_radial = 5
n_axial = 2
radius = .5
height = 5


theta = np.linspace(0, 2 * np.pi, n_radial, endpoint=False)  # don't include 2pi
pts = np.stack([radius * np.cos(theta), radius * np.sin(theta), 0 * theta], axis=1)
tm = TubeMesher(closed=False, triangulate=True, flip_faces=False)

for z in np.linspace(-height / 2, height / 2, n_axial):
    pts[:, 2] = z
    tm.add_xs(t=z, theta=theta, pts=pts)

# sm = tm.mesh.copy()
# m = sm.to_pyvista(True)
# e = tm.mesh.to_pyvista_edges(True)
#

def calculator(t, theta):
    return Point3(radius * np.cos(theta), radius * np.sin(theta), t)

tm.tube_mesher.split_long_edges(1, calculator)
sm = tm.finish()

p = Plotter()
# p.add_mesh(e, line_width=True, scalars='edge_idx')
# p.add_point_labels(e.cell_centers().points, [str(i) for i in e.cell_data['edge_idx'] * 2])
p.add_mesh(sm.pv, show_edges=True, scalars='t')
p.show()
