from isaacsim.core.prims import Articulation
art = Articulation(prim_paths_expr="/World/go1/root_joint")
art.initialize()
print("DOF names:", art.dof_names)
print("Num DOF:", art.num_dof)
