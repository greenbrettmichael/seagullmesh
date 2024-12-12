from numpy import arange
from seagullmesh import Mesh3

def make_meshes():
    _m0 = Mesh3.icosahedron()
    _m0.face_data['idx'] = arange(_m0.n_faces)
    _m1 = Mesh3.pyramid(height=2, radius=1.25, base_center=(0, -.5, -0))
    _m1.face_data['idx'] = arange(_m1.n_faces)
    return _m0, _m1


m0, m1 = make_meshes()
c = m0.corefiner(m1).corefine()
t = c.tracker
print(f'subface_created={t.subface_created}, vertex_added={t.vertex_added}')

m0, m1 = make_meshes()
c = m0.corefiner(m1).union()
t = c.tracker
print(f'subface_created={t.subface_created}, vertex_added={t.vertex_added}')