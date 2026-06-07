import trimesh

mesh = trimesh.load_mesh("3d_data/Tooth36_full.stl")

vertices = mesh.vertices
pcd = trimesh.PointCloud(vertices)

pcd_byte = trimesh.exchange.ply.export_ply(pcd, encoding="ascii")
output_file = open("3d_data/Tooth36_full.ply", "wb+")
output_file.write(pcd_byte)
output_file.close()
