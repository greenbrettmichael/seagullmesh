
from numpy import linspace, pi, stack, cos, sin, arange, zeros
from pyvista import Plotter, PolyData
from seagullmesh import Point3, sgm
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


n_axial_up = 20 * n_axial
n_radial_up = 13 * n_radial

surf = zeros((n_axial_up, n_radial_up, 3), dtype=float)
ts = np.linspace(-height / 2, height / 2, n_axial_up)
thetas = np.linspace(0, 2 * np.pi, n_radial_up, endpoint=False)

for i, z in enumerate(ts):
    for j, theta in enumerate(thetas):
        surf[i, j, :] = [radius * np.cos(theta), radius * np.sin(theta), z]

interp = sgm.tube_mesher.TubeInterpolator(ts, thetas, surf)
pt = interp.interpolate(0, pi / 3)
print(pt)
