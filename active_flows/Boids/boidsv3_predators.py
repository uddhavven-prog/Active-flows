import numpy as np

import numba as nb
from numba import njit

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
import matplotlib
print(matplotlib.get_backend())

# --- parameters (mutable dict so the animation closure sees updates) ---
params = dict(
    w_sep      = 0.05,
    w_ali      = 0.05,
    w_coh      = 0.005,
    vis_range  = 75.0,   # was 'radius', now split into two
    prot_range = 15.0,
    max_speed  = 15.0,
    dt    = 10**(0)
)

DIM = 3
N     = 900
Pred_N = 10 
SIM   = 100*150.0
CELL  = 150.0
DT    = 10**(0)

VISUAL_RANGE    = 75.0   # for alignment + cohesion
PROTECTED_RANGE = 15.0   # for separation only

Grid_arr = np.ones(DIM,dtype=int)*int(SIM/CELL)




poses_boids = np.random.rand(N, DIM) * SIM
vels_boids = np.random.rand(N, DIM)
vels_boids = vels_boids / np.linalg.norm(vels_boids, axis=1, keepdims=True) * params['max_speed']


poses_preds = np.random.rand(Pred_N, DIM) * SIM
vels_preds = np.random.rand(Pred_N, DIM)
vels_preds = vels_preds / np.linalg.norm(vels_preds, axis=1, keepdims=True) * params['max_speed']
# --- numba kernels (same as before) ---



@njit
def NN_points_vec(dims):
    n_points = 3 ** dims
    out = np.empty((n_points, dims), dtype=np.int64)
    
    for i in range(n_points):
        remainder = i
        for d in range(dims - 1, -1, -1):
            out[i, d] = (remainder % 3) - 1
            remainder //= 3
    
    return out


@nb.njit(cache=True)
def build_grid_nd(poses_boids, cell_size, grid_dims):
    n = poses_boids.shape[0]
    D = grid_dims.shape[0]

    strides = np.ones(D, dtype=nb.int64)
    for k in range(1, D):
        strides[k] = strides[k-1] * grid_dims[k-1]

    n_cells = strides[D-1] * grid_dims[D-1]
    head = np.full(n_cells, -1, dtype=nb.int32)
    nxt  = np.full(n,       -1, dtype=nb.int32)

    for i in range(n):
        cell = nb.int64(0)
        for j in range(D):
            coord = int(poses_boids[i, j] / cell_size) % grid_dims[j]
            cell += coord * strides[j]
        nxt[i]     = head[cell]
        head[cell] = i

    return head, nxt, strides


@nb.njit(cache=True)
def boids_force(poses_boids, vels_boids, cell_size, grid_arr,sim, prot_range, vis_range, w_sep, w_ali, w_coh):
    shape = poses_boids.shape

    forces = np.zeros(shape)
    n   = shape[0]
    dim = shape[1]
    head, nxt, strides = build_grid_nd(poses_boids, cell_size, grid_arr)
    offsets  = NN_points_vec(dim)

    for i in range(n):
        cell_coord = np.empty(dim, dtype=nb.int64)
        for j in range(dim):
            cell_coord[j] = int(poses_boids[i, j] / cell_size) % grid_arr[j]

        sep   = np.zeros(dim)
        ali   = np.zeros(dim)
        coh   = np.zeros(dim)
        count = 0

        for oi in range(offsets.shape[0]):
            cell = nb.int64(0)
            for k in range(dim):
                cell += ((cell_coord[k] + offsets[oi, k]) % grid_arr[k]) * strides[k]

            j = head[cell]
            while j != -1:
                ddr = poses_boids[j] - poses_boids[i]

                for k in range(dim):
                    if ddr[k] >  sim * 0.5: ddr[k] -= sim
                    if ddr[k] < -sim * 0.5: ddr[k] += sim

                dist = 0.0
                for k in range(dim):
                    dist += ddr[k] ** 2
                dist = dist ** 0.5

                if dist < prot_range:
                    sep -= ddr
                elif dist < vis_range:
                    ali += vels_boids[j]
                    coh += ddr
                    count += 1

                j = nxt[j]

        if count > 0:
            ali = ali / count - vels_boids[i]
            coh = coh / count

        forces[i] = w_sep*sep + w_ali*ali + w_coh*coh

    return forces

@nb.njit(cache=True)
def preds_force(poses_boids,poses_preds, cell_size, grid_arr,sim, vis_range, w_coh):
    shape = poses_preds.shape
    forces = np.zeros(shape)
    n_preds   = shape[0]
    dim = shape[1]
    head_boids, nxt_boids, strides = build_grid_nd(poses_boids, cell_size, grid_arr)
    offsets  = NN_points_vec(dim)

    for i in range(n_preds):
        cell_coord = np.empty(dim, dtype=nb.int64)
        for j in range(dim):
            cell_coord[j] = int(poses_preds[i, j] / cell_size) % grid_arr[j]

        coh   = np.zeros(dim)
        count = 0

        for oi in range(offsets.shape[0]):
            cell = nb.int64(0)
            for k in range(dim):
                cell += ((cell_coord[k] + offsets[oi, k]) % grid_arr[k]) * strides[k]

            j = head_boids[cell]
            while j != -1:
                ddr = poses_boids[j] - poses_preds[i]

                for k in range(dim):
                    if ddr[k] >  sim * 0.5: ddr[k] -= sim
                    if ddr[k] < -sim * 0.5: ddr[k] += sim

                dist = 0.0
                for k in range(dim):
                    dist += ddr[k] ** 2
                dist = dist ** 0.5

                
                if dist < vis_range:
                    coh += ddr
                    count += 1

                j = nxt_boids[j]

        if count > 0:
            coh = coh / count

        forces[i] = w_coh*coh

    return forces

@nb.njit(cache=True)
def preds_force_on_boids (poses_boids,poses_preds, cell_size, grid_arr,sim,vis_range, w_evr):
    shape = poses_boids.shape

    forces = np.zeros(shape)
    n   = shape[0]
    dim = shape[1]
    head_preds, nxt_preds, strides = build_grid_nd(poses_preds, cell_size, grid_arr)
    offsets  = NN_points_vec(dim)

    for i in range(n):
        cell_coord = np.empty(dim, dtype=nb.int64)
        for j in range(dim):
            cell_coord[j] = int(poses_boids[i, j] / cell_size) % grid_arr[j]

        evr   = np.zeros(dim)
        count = 0

        for oi in range(offsets.shape[0]):
            cell = nb.int64(0)
            for k in range(dim):
                cell += ((cell_coord[k] + offsets[oi, k]) % grid_arr[k]) * strides[k]

            j = head_preds[cell]
            while j != -1:
                ddr = poses_preds[j] - poses_boids[i]

                for k in range(dim):
                    if ddr[k] >  sim * 0.5: ddr[k] -= sim
                    if ddr[k] < -sim * 0.5: ddr[k] += sim

                dist = 0.0
                for k in range(dim):
                    dist += ddr[k] ** 2
                dist = dist ** 0.5

                
                if dist < vis_range:
                    evr -= ddr
                    count += 1

                j = nxt_preds[j]

        if count > 0:
            evr = evr / count

        forces[i] =  w_evr*evr

    return forces

@nb.njit(cache=True)
def pbc_and_speedcap(poses,vels,sim,maxspeed,dt):
    n = np.shape(poses)[0]
    dim = np.shape(poses)[1]

    for i in range(n):

        spd = 0
        for j in range(dim):
            spd+= vels[i,j]**2
        spd = np.sqrt(spd)

        if spd > maxspeed:
            vels[i] = vels[i]*maxspeed/spd
    poses = (poses + vels * dt) % sim
    return poses , vels


@nb.njit(cache=True)
def update_boids(poses_boids, vels_boids,poses_preds, vels_preds, cell_size, grid_arr,sim, 
                 prot_range, vis_range,
                 maxspeed,dt,
                   w_sep, w_ali, w_coh,w_evr):
    f_boids = boids_force(poses_boids, vels_boids, cell_size, grid_arr,sim, prot_range, vis_range, w_sep, w_ali, w_coh)
    f_preds = preds_force(poses_boids, poses_preds, cell_size, grid_arr,sim, vis_range, w_coh)
    f_pred_on_boids = preds_force_on_boids(poses_boids,poses_preds,cell_size,grid_arr,sim,vis_range,w_evr)
    new_poses_boids, new_vels_boids = pbc_and_speedcap(poses_boids,vels_boids+f_boids+f_pred_on_boids,sim,maxspeed,dt)
    new_poses_preds, new_vels_preds = pbc_and_speedcap(poses_preds,vels_preds+f_preds,sim,maxspeed,dt)

    return new_poses_boids, new_vels_boids,new_poses_preds, new_vels_preds


params = dict(
    w_sep      = 0.05,
    w_ali      = 0.05,
    w_coh      = 0.005,
    vis_range  = 75.0,   # was 'radius', now split into two
    prot_range = 15.0,
    max_speed  = 15.0,
    dt    = 10**(0),
    w_evr=5.0
)


# --- warmup ---
poses_boids, vels_boids,poses_preds, vels_preds = update_boids(poses_boids, vels_boids,poses_preds, vels_preds, CELL, Grid_arr, SIM,
                            params['prot_range'],params['vis_range'],
                              params['max_speed'], DT,
                            params['w_sep'], params['w_ali'], params['w_coh'], params['w_evr'])





slider_vars = ['prot_range','vis_range','max_speed','dt','w_sep','w_ali', 'w_coh','w_evr']
slider_min =  [0.0          ,0.0       ,0.0        ,0.0 ,0.0    ,0.0    ,0.0     ,0.0  ]
slider_max =  [2*CELL       ,2*CELL    ,50.0       ,10.0 ,1.0   ,1.0    ,0.1     ,10.0]



#Setup Figure

if DIM ==2:
    fig, ax = plt.subplots()
    ax.set_xlim(0, SIM)
    ax.set_ylim(0, SIM)
    ax.set_aspect('equal')
    #ax.set_position([0.1, 0.15, 0.8, 0.8])
    ax.set_position([0.1, 0.42, 0.8, 0.55])   # 2D case
    #sc = ax.scatter(poses_boids[:, 0], poses_boids[:, 1], s=20)

    sc_boid = ax.quiver(poses_boids[:, 0], poses_boids[:, 1],   # positions
                   vels_boids[:, 0],  vels_boids[:, 1],    # directions
                   angles='xy', scale_units='xy', color='steelblue',scale=0.001*int(SIM/CELL),headwidth=4, headlength=5, headaxislength=4,width=0.003)
    sc_pred = ax.quiver(poses_preds[:, 0], poses_preds[:, 1],   # positions
                   vels_preds[:, 0],  vels_preds[:, 1],    # directions
                   angles='xy', scale_units='xy',color='red', scale=int(SIM/CELL),headwidth=1, headlength=5, headaxislength=4,width=0.003)
elif DIM == 3:
    fig = plt.figure()
    ax  = fig.add_subplot(projection='3d')
    ax.set_xlim(0, SIM)
    ax.set_ylim(0, SIM)
    ax.set_zlim(0, SIM)

    #fig.subplots_adjust(bottom=0.40)
    fig.subplots_adjust(bottom=0.40, top=0.98)
    sc_boid = ax.scatter(poses_boids[:, 0], poses_boids[:, 1], poses_boids[:, 2], s=20)
    sc_pred = ax.scatter(poses_preds[:, 0], poses_preds[:, 1], poses_preds[:, 2], s=20)



#(poses_boids, vels_boids, cell_size, grid_arr,sim, prot_range, vis_range, max_speed, dt, w_sep, w_ali, w_coh)
#params['prot_range'],params['vis_range'],params['max_speed'], DT,params['w_sep'], params['w_ali'], params['w_coh']

#code needs to be updated all slider are in same location


slider_dict = {}
slider_spacing = 0.05
for i in range(len(slider_vars)):
    ax_S = fig.add_axes([0.2, 0.02 + i * slider_spacing, 0.6, 0.03])
    #ax_S = fig.add_axes([0.2, 0.06, 0.6, 0.03])
    slider_i = Slider(ax_S,  slider_vars[i],  slider_min[i], slider_max[i],  valinit=params[slider_vars[i]])
    slider_dict[slider_vars[i]] = slider_i
slider_numb = len(slider_vars)


def frame2d(i):
    global poses_boids, vels_boids,poses_preds,vels_preds,CELL, Grid_arr, SIM
    #poses_boids, vels_boids = update_boids(poses_boids, vels_boids, CELL,Grid_arr,SIM,s_prot_range.val,s_vis_range.val,s_max_speed.val,s_dt.val,s_w_sep.val,s_w_ali.val,s_w_coh.val)
    poses_boids, vels_boids, poses_preds, vels_preds= update_boids(poses_boids, vels_boids,poses_preds, vels_preds, CELL,Grid_arr,SIM,slider_dict['prot_range'].val,
                               slider_dict['vis_range'].val,slider_dict['max_speed'].val,slider_dict['dt'].val,
                               slider_dict['w_sep'].val,slider_dict['w_ali'].val,slider_dict['w_coh'].val,slider_dict['w_evr'].val)
    #sc.set_offsets(poses_boids)


    sc_boid.set_offsets(np.c_[poses_boids[:, 0], poses_boids[:, 1]])  # update positions
    sc_pred.set_offsets(np.c_[poses_preds[:, 0], poses_preds[:, 1]]) #need to change colour
    
    norms = np.linalg.norm(vels_boids, axis=1, keepdims=True)
    unit_vels_boids = vels_boids / norms
    norms = np.linalg.norm(vels_preds, axis=1, keepdims=True)
    unit_vels_preds = vels_preds / norms
    
    sc_boid.set_UVC(unit_vels_boids[:, 0], unit_vels_boids[:, 1])               # update directions
    sc_pred.set_UVC(unit_vels_preds[:, 0], unit_vels_preds[:, 1])
    return (sc_boid, sc_pred)

def frame3d(i):    
    global poses_boids, vels_boids,poses_preds,vels_preds,CELL, Grid_arr, SIM
    #poses_boids, vels_boids = update_boids(poses_boids, vels_boids, CELL,Grid_arr,SIM,s_prot_range.val,s_vis_range.val,s_max_speed.val,s_dt.val,s_w_sep.val,s_w_ali.val,s_w_coh.val)
    poses_boids, vels_boids, poses_preds, vels_preds= update_boids(poses_boids, vels_boids,poses_preds, vels_preds, CELL,Grid_arr,SIM,slider_dict['prot_range'].val,
                               slider_dict['vis_range'].val,slider_dict['max_speed'].val,slider_dict['dt'].val,
                               slider_dict['w_sep'].val,slider_dict['w_ali'].val,slider_dict['w_coh'].val,slider_dict['w_evr'].val)
    #sc.set_offsets(poses_boids)


  
    sc_boid._offsets3d = (poses_boids[:, 0], poses_boids[:, 1], poses_boids[:, 2])
    sc_pred._offsets3d = (poses_preds[:, 0], poses_preds[:, 1], poses_preds[:, 2])
    return (sc_boid, sc_pred)

if DIM ==2:
    ani = animation.FuncAnimation(fig, frame2d, interval=30, blit=True)

elif DIM == 3:
    ani = animation.FuncAnimation(fig, frame3d, interval=30, blit=False)
plt.show()