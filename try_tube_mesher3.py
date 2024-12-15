from seagullmesh.tube_mesher import TubeMesher

# cyl = TubeMesher.cylinder(closed=False, flip_faces=False, n_axial=2, n_radial=3)
cyl = TubeMesher.cylinder(closed=False, flip_faces=False, n_axial=2, n_radial=3, triangulate=True)
pts, faces = cyl.to_polygon_soup()
print(faces)
