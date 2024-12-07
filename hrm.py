from seagullmesh import Mesh3, sgm, Point2
from numpy import arange, ones, zeros

m = Mesh3.icosahedron()
idx = sgm.mesh.Vertices(list(set(m.vertices)))
