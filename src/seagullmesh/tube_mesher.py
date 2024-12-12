from __future__ import annotations

import numpy as np

from seagullmesh import Mesh3


class TubeMesher:
    def __init__(self, t0: float, theta0: np.ndarray, pts0: np.ndarray, closed=False):
        mesh = self.mesh = Mesh3()
        t_map = mesh.vertex_data.add_property('t', default=-1.0)
        theta_map = mesh.vertex_data.add_property('theta', default=-1.0)
        self.tube_mesher = sgm.triangulate.TubeMesher(mesh.mesh, t_map.pmap, theta_map.pmap, t0, theta0, pts0)

        self.closed = closed
        if closed:
            self.tube_mesher.close_xs(False)

    def add_xs(self, t: float, theta: np.ndarray, pts: np.ndarray):
        self.tube_mesher.add_xs(t, theta, pts)

    def finish(self, reverse_orientation: bool):
        if self.closed:
            self.tube_mesher.close_xs(True)

        sgm.triangulate.triangulate_faces(self.mesh.mesh, self.mesh.faces)
        self.mesh.collect_garbage()
        if reverse_orientation:
            sgm.triangulate.reverse_face_orientations(self.mesh.mesh, self.mesh.faces)

        return self.mesh
