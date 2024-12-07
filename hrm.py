from seagullmesh import Mesh3, sgm, Point2
from numpy import arange, ones, zeros

m = Mesh3.icosahedron()
pmap = m.vertex_data.add_property('p', default=Point2(0, 0))
# print(type(pmap), type(pmap.pmap))
print(pmap[m.null_face])
