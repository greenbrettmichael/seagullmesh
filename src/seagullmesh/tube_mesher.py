from __future__ import annotations

import numpy as np

from seagullmesh import Mesh3
from seagullmesh._seagullmesh import tube_mesher


class TubeMesher:
    def __init__(self, closed=False):
        mesh = self.mesh = Mesh3()
        t_map = mesh.vertex_data.add_property('t', default=-1.0)
        theta_map = mesh.vertex_data.add_property('theta', default=-1.0)
        is_cap_map = mesh.face_data.add_property('is_cap', default=False)
        self.tube_mesher = tube_mesher.TubeMesher(
            mesh.mesh, t_map.pmap, theta_map.pmap, is_cap_map.pmap)
        self.closed = closed

    def add_xs(self, t: float, theta: np.ndarray, pts: np.ndarray):
        self.tube_mesher.add_xs(t, theta, pts)

        if self.closed and self.tube_mesher.nxs == 0:
            self.tube_mesher.close_xs(False)

    def finish(self, reverse_orientation: bool):
        if self.closed:
            self.tube_mesher.close_xs(True)

        sgm.triangulate.triangulate_faces(self.mesh.mesh, self.mesh.faces)
        if reverse_orientation:
            sgm.triangulate.reverse_face_orientations(self.mesh.mesh, self.mesh.faces)

        return self.mesh
