from seagullmesh import Mesh3

m0 = Mesh3.icosahedron()
m1 = Mesh3.pyramid(height=2, radius=1.25, base_center=(0, -.5, -0))

m0.corefiner(m1)

# ).to_pyvista().plot(show_edges=True)