from numpy import array, cos, sin
from seagullmesh import Mesh3


def tetrahedron(scale=1.0, rot_z=0.0):
    verts = scale * array([[1, 1, 1], [-1, 1, -1], [1, -1, -1], [-1, -1, 1]], dtype='float')

    if rot_z:
        rot = array([[cos(rot_z), -sin(-rot_z), 0], [sin(rot_z), cos(-rot_z), 0], [0, 0, 1]])
        verts = verts @ rot.T

    faces = array([[2, 1, 0], [2, 3, 1], [3, 2, 0], [1, 3, 0]], dtype='int')
    return verts, faces


def tetrahedron_mesh() -> Mesh3:
    return Mesh3.from_polygon_soup(*tetrahedron())


m = tetrahedron_mesh()
foo = m.halfedge_data.create('foo', default=3)
foo[m.halfedges] = 3
print(m.edges, foo[m.halfedges])