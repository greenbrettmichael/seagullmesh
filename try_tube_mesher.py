
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

# TODO: default behavior is flipped?
sm = TubeMesher.cylinder(n_radial=5, n_axial=2, closed=False)

theta = np.linspace(0, 2 * np.pi, n_radial, endpoint=False)  # don't include 2pi
pts = np.stack([radius * np.cos(theta), radius * np.sin(theta), 0 * theta], axis=1)
tm = TubeMesher(closed=False)

for z in np.linspace(-height / 2, height / 2, n_axial):
    pts[:, 2] = z
    tm.add_xs(t=z, theta=theta, pts=pts)


def calculator(t, theta):
    return Point3(radius * np.cos(theta), radius * np.sin(theta), t)


tm.tube_mesher.split_long_edges(1, calculator)

tm.finish().pv.plot(show_edges=True)
