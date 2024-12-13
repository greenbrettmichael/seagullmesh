from __future__ import annotations

import numpy as np

from seagullmesh import Mesh3
from seagullmesh._seagullmesh import tube_mesher


# TODO tests: assert does_bound_a_volume


class TubeMesher:
    def __init__(self, closed=False, triangulate=True):
        mesh = self.mesh = Mesh3()
        t_map = mesh.vertex_data.add_property('t', default=-1.0)
        theta_map = mesh.vertex_data.add_property('theta', default=-1.0)
        is_cap_map = mesh.face_data.add_property('is_cap', default=False)
        self.tube_mesher = tube_mesher.TubeMesher(
            mesh.mesh, t_map.pmap, theta_map.pmap, is_cap_map.pmap, triangulate)
        self.closed = closed
        self.triangulate = triangulate

    def add_xs(self, t: float, theta: np.ndarray, pts: np.ndarray):
        self.tube_mesher.add_xs(t, theta, pts)

        if self.closed and self.tube_mesher.nxs == 1:  # i.e. this was the first xs
            self.tube_mesher.close_xs(False)

    def finish(self, reverse_orientation: bool = False) -> Mesh3:
        if self.closed:
            self.tube_mesher.close_xs(True)

        if self.triangulate:
            self.mesh.collect_garbage()  # clear untriangulated faces

        if reverse_orientation:
            self.mesh.reverse_face_orientations()

        return self.mesh

    @staticmethod
    def cylinder(
            n_radial: int = 5,
            n_axial: int = 5,
            closed: bool = False,
            height: float = 1.0,
    ) -> Mesh3:
        theta = np.linspace(0, 2 * np.pi, n_radial)
        pts = np.stack([np.cos(theta), np.sin(theta), 0 * theta], axis=1)
        tm = TubeMesher(closed=closed)

        for z in np.linspace(0, height, n_axial, endpoint=True):
            pts[:, 2] = z
            tm.add_xs(t=z, theta=theta, pts=pts)

        return tm.finish()
