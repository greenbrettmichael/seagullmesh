import numpy as np
from numpy import array, zeros
from pyvista import Plotter

from seagullmesh import Mesh3
from seagullmesh.tube_mesher import TubeMesher
from seagullmesh.util import axis_angle

sm0 = TubeMesher.cylinder(radius=0.75, n_radial=5, n_axial=5, closed=True, flip_faces=True, height=5)
sm1 = TubeMesher.cylinder(radius=0.5, n_radial=7, n_axial=7, closed=True, flip_faces=True, height=5)
sm1.transform(axis_angle((1, 0, 0), np.pi / 2))
sms = [sm0, sm1]

for i, sm in enumerate(sms):
    sm.face_data['source'] = i
    sm.face_data['idx'] = np.arange(sm.n_faces)

orig = [sm.to_pyvista(True) for sm in sms]
corefined = sm0.corefiner(sm1).track(face_mesh_map='face_mesh_map', face_face_map='face_face_map').union()
corefined.update_face_properties(0, property_names=('is_cap',))

output = sm0.to_pyvista(data=True)


sm0.to_pyvista(True).plot(scalars='is_cap', show_edges=True)


# p = Plotter(shape=(2, 2))
#
# for i, (orig_prop, coref_prop) in enumerate([('source', 'face_origin'), ('idx', 'face_idx')]):
#     p.subplot(i, 0)
#     for m in orig:
#         p.add_mesh(m.copy(), show_edges=True, scalars=orig_prop)
#
#     p.subplot(i, 1)
#     p.add_mesh(output.copy(), show_edges=True, scalars=coref_prop)
# p.link_views()
# p.show()
