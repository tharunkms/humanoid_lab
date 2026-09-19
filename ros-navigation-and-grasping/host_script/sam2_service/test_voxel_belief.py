import numpy as np, sys, time
sys.path.insert(0,'/home/claude/step1')
from voxel_belief import VoxelBelief
W,H=640,360; fx=fy=460.0; cx,cy=W/2,H/2
table_z=0.0; ctr=np.array([0.0,0.0,table_z]); R=0.45
box_lo=np.array([-0.035,-0.025,0.0]); box_hi=np.array([0.035,0.025,0.06])

def render(cam_pos, look_at):
    f=np.asarray(look_at,float)-cam_pos; f/=np.linalg.norm(f)
    up=np.array([0,0,1.0]); r=np.cross(f,up); r/=np.linalg.norm(r); u=np.cross(r,f)
    Rm=np.stack([r,-u,f],axis=1)
    T=np.eye(4); T[:3,:3]=Rm; T[:3,3]=cam_pos
    u_,v_=np.meshgrid(np.arange(W),np.arange(H))
    dc=np.stack([(u_-cx)/fx,(v_-cy)/fy,np.ones_like(u_,float)],-1)  # camera frame, z=1 plane
    dw=dc@Rm.T                                                       # world, NOT normalised: t is then the pinhole Z
    o=cam_pos
    with np.errstate(divide='ignore',invalid='ignore'):
        t_tab=np.where(dw[...,2]<-1e-9,(table_z-o[2])/dw[...,2],np.inf)
    p=o+dw*t_tab[...,None]
    t_tab=np.where(np.linalg.norm(p[...,:2]-ctr[:2],axis=-1)<=R,t_tab,np.inf)
    t0=np.full((H,W),-np.inf); t1=np.full((H,W),np.inf)
    for i in range(3):
        with np.errstate(divide='ignore',invalid='ignore'):
            ta=(box_lo[i]-o[i])/dw[...,i]; tb=(box_hi[i]-o[i])/dw[...,i]
        t0=np.maximum(t0,np.minimum(ta,tb)); t1=np.minimum(t1,np.maximum(ta,tb))
    t_box=np.where((t0<=t1)&(t1>0),np.where(t0>0,t0,t1),np.inf)
    t=np.minimum(t_tab,t_box); t[~np.isfinite(t)]=0
    return np.where(t>0,t,0).astype(np.float32), T   # t IS the pinhole Z here

g=VoxelBelief.around_table(ctr,R,table_z,res=0.005,height=0.4)
cam=np.array([0.35,-0.35,0.30])
depth,T=render(cam,np.array([0,0,0.03]))
print(f"depth: {(depth>0).mean():.0%} valid, range {depth[depth>0].min():.2f}..{depth[depth>0].max():.2f} m")
t0=time.time(); n=g.integrate_depth(depth,T,fx,fy,cx,cy,stride=2); dt=time.time()-t0
print(f"integrated {n} rays in {dt*1000:.0f} ms"); print(g.stats())
occ=g.occupied_points(); print(f"\noccupied voxels: {len(occ)}")
on_table=np.abs(occ[:,2]-table_z)<0.008
in_box=np.all((occ>=box_lo-0.008)&(occ<=box_hi+0.008),axis=1)
print(f"  on table plane {on_table.mean():.1%}   on/in box {in_box.mean():.1%}   elsewhere {(~on_table&~in_box).mean():.1%}")
if in_box.any(): print(f"  box voxel z range {occ[in_box][:,2].min():.3f}..{occ[in_box][:,2].max():.3f} (true 0..0.060)")
def obs(p):
    i=g.world_to_idx(np.array([p]))
    return bool(g.observed[tuple(i[0])]) if g.in_bounds(i)[0] else None
print(f"  near-side surface observed {obs([0.037,-0.027,0.03])}   far-side {obs([-0.037,0.027,0.03])} (want False)")
print(f"  inside box {obs([0,0,0.03])} (want False)   above box {obs([0,0,0.12])} (want True)")
unk=g.unknown_near(np.array([0,0,0.03]),radius=0.12,exclude_below_z=table_z)
print(f"\nunknown within 12 cm of box centre: {len(unk)}")
if len(unk): print(f"  centroid {np.round(unk.mean(0),3)}  (camera at {cam} -> expect opposite side: x<0, y>0)")

# ---- multi-view accumulation ----
print("\n--- 4 views around the box ---")
g2=VoxelBelief.around_table(ctr,R,table_z,res=0.005,height=0.4)
import time
for k,(cx_,cy_) in enumerate([(0.35,-0.35),(-0.35,-0.35),(-0.35,0.35),(0.35,0.35)]):
    c=np.array([cx_,cy_,0.30]); d,T=render(c,np.array([0,0,0.03]))
    t=time.time(); g2.integrate_depth(d,T,fx,fy,cx,cy,stride=3); print(f"  view {k}: {(time.time()-t)*1000:.0f} ms  {g2.stats()}")
occ=g2.occupied_points(); in_box=np.all((occ>=box_lo-0.008)&(occ<=box_hi+0.008),axis=1)
print(f"box voxels now {int(in_box.sum())} (1 view had ~240)")
unk=g2.unknown_near(np.array([0,0,0.03]),radius=0.12,exclude_below_z=table_z)
print(f"unknown within 12 cm after 4 views: {len(unk)} (was 4282 after 1)")
bb=occ[in_box]; print(f"box extent from belief: {np.round(bb.max(0)-bb.min(0),3)} m (true 0.070 0.050 0.060)")
