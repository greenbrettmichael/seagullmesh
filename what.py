from seagullmesh import sgm, Mesh3
m = Mesh3.icosahedron()
foo = m.face_data.add_property('foo', default=False)
fs = m.faces
print(fs[0], foo[fs[0]])
print(foo.pmap.get_vector(fs.indices))
print(foo.pmap[fs.indices])
# print(foo.get_vector(fs))
# print(foo[fs], foo[fs[0]])
