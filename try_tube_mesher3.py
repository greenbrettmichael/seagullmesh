import numpy as np
from pyvista import Plotter

from seagullmesh import Mesh3
from seagullmesh.tube_mesher import TubeMesher


def irregular(
        n_radial: int = 15,
        n_axial: int = 5,
        radius: float = 1.0,
        height: float = 1.0,
        n_extra_range: tuple[int, int] = (1, 5),
        **kwargs
) -> Mesh3:
    theta = np.linspace(0, 2 * np.pi, n_radial, endpoint=False)  # don't include 2pi

    tm = TubeMesher(**kwargs)

    for z in np.linspace(-height / 2, height / 2, n_axial):
        n_extra = np.random.randint(n_extra_range[0], n_extra_range[1])
        theta1 = np.concat([theta, np.random.uniform(0, 2 * np.pi, size=n_extra)])
        theta1.sort()
        print(theta1)

        pts = np.stack([
            radius * np.cos(theta1),
            radius * np.sin(theta1),
            z * np.ones_like(theta1),
        ], axis=1)

        tm.add_xs(t=z, theta=theta1, pts=pts)

    return tm.finish()

# sm = TubeMesher.cylinder(closed=False, flip_faces=False, n_axial=2, n_radial=3)
sm = irregular(
    closed=True, flip_faces=False, n_axial=10, n_radial=20, triangulate=True, n_extra_range=(10, 20))

# print(sm.n_faces)
m = sm.to_pyvista(True)
p = Plotter()
p.add_mesh(m, show_edges=True, opacity=1)
p.show()