import numpy as np
from numpy import array, zeros
from pyvista import Plotter

from seagullmesh import Mesh3
from seagullmesh.tube_mesher import TubeMesher
from seagullmesh.util import axis_angle

# sm0 = TubeMesher.cylinder(radius=0.75, n_radial=5, n_axial=5, closed=True, flip_faces=True, height=5)
# sm1 = TubeMesher.cylinder(radius=0.5, n_radial=7, n_axial=7, closed=True, flip_faces=True, height=5)

sm0 = Mesh3.icosahedron()
sm1 = Mesh3.icosahedron(center=(.1, .1, .1))

sm1.transform(axis_angle((1, 0, 0), np.pi / 2))
sms = [sm0, sm1]

for i, sm in enumerate(sms):
    sm.face_data['source'] = i
    sm.face_data['idx'] = np.arange(sm.n_faces)

orig = [sm.to_pyvista(True) for sm in sms]
corefined = sm0.corefiner(sm1).track(face_mesh_map='face_mesh_map', face_face_map='face_face_map').corefine()

print((sm0.face_data['face_mesh_map'][:] == -1).all())
corefined.update_face_properties(0, property_names=('idx',))

output = sm0.to_pyvista(face_data=('idx', 'source', 'face_mesh_map'))

output.plot(scalars='idx', show_edges=True)

# p = Plotter(shape=(1, 2))
#
# for i, m in enumerate((orig[0], output)):
#     p.subplot(0, i)
#     p.add_mesh(m, show_edges=True, scalars='face_mesh_map')
#
# p.link_views()
# p.show()
