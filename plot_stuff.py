from seagullmesh import Mesh3

m1 = Mesh3.icosahedron().add(
    Mesh3.pyramid(height=2, radius=1.25, base_center=(0, -.5, -0))
).to_pyvista().plot(show_edges=True)