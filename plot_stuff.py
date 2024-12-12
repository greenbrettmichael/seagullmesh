from numpy import arange

from seagullmesh import Mesh3
m0 = Mesh3.icosahedron()
m0.face_data['idx'] = arange(m0.n_faces)
m0.face_data['foo'] = 1
m1 = Mesh3.pyramid(height=2, radius=1.25, base_center=(0, -.5, -0))
m1.face_data['foo'] = 2
m1.face_data['idx'] = arange(m1.n_faces)
c = m0.corefiner(m1).corefine()
c.update_split_faces(0)
# corefined.update_copied_faces()


# vs = corefined.get_new_vertices(0)
# corefined.label_new_vertices(0, 'is_new')

m0.to_pyvista(face_data='all').plot(show_edges=True, scalars='idx')
