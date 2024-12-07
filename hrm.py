from seagullmesh import Mesh3, sgm, Point2
from numpy import arange, ones, zeros

m = Mesh3.icosahedron()
idxs = m.vertices
idxs[1:] = idxs[1:][::-1]
