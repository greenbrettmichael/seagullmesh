from __future__ import annotations

import numpy as np

from seagullmesh import Mesh3, PropertyMap, Vertex, Face
from seagullmesh._seagullmesh import tube_mesher


# TODO tests: assert does_bound_a_volume


class TubeMesher:
    def __init__(
            self,
            closed: bool = False,
            triangulate: bool = True,
            flip_faces: bool = False,
    ):
        mesh = self.mesh = Mesh3()
        t_map = mesh.vertex_data.create('t', default=-1.0)
        theta_map = mesh.vertex_data.create('theta', default=-1.0)
        is_cap_map = mesh.face_data.create('is_cap', default=False)
        self.tube_mesher = tube_mesher.TubeMesher(
            mesh.mesh, t_map.pmap, theta_map.pmap, is_cap_map.pmap,
            closed, triangulate, flip_faces)

    def add_xs(self, t: float, theta: np.ndarray, pts: np.ndarray):
        self.tube_mesher.add_xs(t, theta, pts)

    def finish(self) -> Mesh3:
        self.tube_mesher.finish()
        return self.mesh

    @staticmethod
    def cylinder(
            n_radial: int = 5,
            n_axial: int = 5,
            closed: bool = False,
            radius: float = 1.0,
            height: float = 1.0,
            flip_faces: bool = False,
            **kwargs
    ) -> Mesh3:
        theta = np.linspace(0, 2 * np.pi, n_radial, endpoint=False)  # don't include 2pi
        pts = np.stack([radius * np.cos(theta), radius * np.sin(theta), 0 * theta], axis=1)
        tm = TubeMesher(closed=closed, flip_faces=flip_faces, **kwargs)

        for z in np.linspace(-height / 2, height / 2, n_axial):
            pts[:, 2] = z
            tm.add_xs(t=z, theta=theta, pts=pts)

        return tm.finish()
