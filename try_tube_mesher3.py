from pyvista import Plotter

from seagullmesh.tube_mesher import TubeMesher

# cyl = TubeMesher.cylinder(closed=False, flip_faces=False, n_axial=2, n_radial=3)
sm = TubeMesher.cylinder(closed=True, flip_faces=False, n_axial=3, n_radial=3, triangulate=True)

pts, faces = sm.to_polygon_soup()
print(faces)
m = sm.to_pyvista(True)
normals = m.compute_normals().cell_data['Normals']

p = Plotter()
p.add_mesh(m, show_edges=True, opacity=0.5)
p.add_arrows(m.cell_centers().points, normals * 0.5)
p.show()