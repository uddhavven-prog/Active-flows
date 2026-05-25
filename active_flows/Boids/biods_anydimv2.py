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

DIM = 2
N     = 900
SIM   = 100*150.0
CELL  = 150.0
DT    = 10**(0)

VISUAL_RANGE    = 75.0   # for alignment + cohesion
PROTECTED_RANGE = 15.0   # for separation only

Grid_arr = np.ones(DIM,dtype=int)*int(SIM/CELL)




poses = np.random.rand(N, DIM) * SIM
vels = np.random.rand(N, DIM)
vels = vels / np.linalg.norm(vels, axis=1, keepdims=True) * params['max_speed']
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
def build_grid_nd(poses, cell_size, grid_dims):
    n = poses.shape[0]
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
            coord = int(poses[i, j] / cell_size) % grid_dims[j]
            cell += coord * strides[j]
        nxt[i]     = head[cell]
        head[cell] = i

    return head, nxt, strides


@nb.njit(cache=True)
def update_boids(poses, vels, cell_size, grid_arr,sim, prot_range, vis_range, max_speed, dt, w_sep, w_ali, w_coh):
    n   = poses.shape[0]
    dim = poses.shape[1]
    head, nxt, strides = build_grid_nd(poses, cell_size, grid_arr)
    new_vels = vels.copy()
    offsets  = NN_points_vec(dim)

    for i in range(n):
        cell_coord = np.empty(dim, dtype=nb.int64)
        for j in range(dim):
            cell_coord[j] = int(poses[i, j] / cell_size) % grid_arr[j]

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
                ddr = poses[j] - poses[i]

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
                    ali += vels[j]
                    coh += ddr
                    count += 1

                j = nxt[j]

        if count > 0:
            ali = ali / count - vels[i]
            coh = coh / count

        s = w_sep*sep + w_ali*ali + w_coh*coh
        new_vels[i] = vels[i] + s * dt

        #give some biods random kicks
        percent = 0.01
        if percent < np.random.uniform():
            new_vels[i] = 2*new_vels[i]    
        
        spd = 0.0
        for k in range(dim):
            spd += new_vels[i, k] ** 2
        spd = spd ** 0.5

        if spd > max_speed:
            new_vels[i] = new_vels[i] / spd * max_speed

    new_poses = (poses + new_vels * dt) % sim
    return new_poses, new_vels
# --- warmup ---
poses, vels = update_boids(poses, vels, CELL, Grid_arr, SIM,
                            params['prot_range'],params['vis_range'],
                              params['max_speed'], DT,
                            params['w_sep'], params['w_ali'], params['w_coh'])

#Setup Figure

if DIM ==2:
    fig, ax = plt.subplots()
    ax.set_xlim(0, SIM)
    ax.set_ylim(0, SIM)
    ax.set_aspect('equal')
    #ax.set_position([0.1, 0.15, 0.8, 0.8])
    ax.set_position([0.1, 0.42, 0.8, 0.55])   # 2D case
    #sc = ax.scatter(poses[:, 0], poses[:, 1], s=20)

    sc = ax.quiver(poses[:, 0], poses[:, 1],   # positions
                   vels[:, 0],  vels[:, 1],    # directions
                   angles='xy', scale_units='xy', scale=0.05*int(SIM/CELL),headwidth=4, headlength=5, headaxislength=4,width=0.003)
elif DIM == 3:
    fig = plt.figure()
    ax  = fig.add_subplot(projection='3d')
    ax.set_xlim(0, SIM)
    ax.set_ylim(0, SIM)
    ax.set_zlim(0, SIM)

    #fig.subplots_adjust(bottom=0.40)
    fig.subplots_adjust(bottom=0.40, top=0.98)
    sc = ax.scatter(poses[:, 0], poses[:, 1], poses[:, 2], s=20)


#(poses, vels, cell_size, grid_arr,sim, prot_range, vis_range, max_speed, dt, w_sep, w_ali, w_coh)
#params['prot_range'],params['vis_range'],params['max_speed'], DT,params['w_sep'], params['w_ali'], params['w_coh']

#code needs to be updated all slider are in same location

params = dict(
    w_sep      = 0.05,
    w_ali      = 0.05,
    w_coh      = 0.005,
    vis_range  = 75.0,   # was 'radius', now split into two
    prot_range = 15.0,
    max_speed  = 15.0,
    dt    = 10**(0)
)



slider_vars = ['prot_range','vis_range','max_speed','dt','w_sep','w_ali', 'w_coh']
slider_min =  [0.0          ,0.0       ,0.0        ,0.0 ,0.0    ,0.0    ,0.0     ]
slider_max =  [2*CELL       ,2*CELL    ,50.0       ,10.0 ,1.0   ,1.0    ,0.1     ]
slider_dict = {}
slider_spacing = 0.05
for i in range(len(slider_vars)):
    ax_S = fig.add_axes([0.2, 0.02 + i * slider_spacing, 0.6, 0.03])
    #ax_S = fig.add_axes([0.2, 0.06, 0.6, 0.03])
    slider_i = Slider(ax_S,  slider_vars[i],  slider_min[i], slider_max[i],  valinit=params[slider_vars[i]])
    slider_dict[slider_vars[i]] = slider_i
slider_numb = len(slider_vars)


def frame2d(i):
    global poses, vels,CELL, Grid_arr, SIM
    #poses, vels = update_boids(poses, vels, CELL,Grid_arr,SIM,s_prot_range.val,s_vis_range.val,s_max_speed.val,s_dt.val,s_w_sep.val,s_w_ali.val,s_w_coh.val)
    poses, vels = update_boids(poses, vels, CELL,Grid_arr,SIM,slider_dict['prot_range'].val,
                               slider_dict['vis_range'].val,slider_dict['max_speed'].val,slider_dict['dt'].val,
                               slider_dict['w_sep'].val,slider_dict['w_ali'].val,slider_dict['w_coh'].val)
    #sc.set_offsets(poses)

    sc.set_offsets(np.c_[poses[:, 0], poses[:, 1]])  # update positions

    norms = np.linalg.norm(vels, axis=1, keepdims=True)
    unit_vels = vels / norms
    sc.set_UVC(unit_vels[:, 0], unit_vels[:, 1])               # update directions
    return (sc,)

def frame3d(i):
    global poses, vels,CELL, Grid_arr, SIM
    #poses, vels = update_boids(poses, vels, CELL,Grid_arr,SIM,s_prot_range.val,s_vis_range.val,s_max_speed.val,s_dt.val,s_w_sep.val,s_w_ali.val,s_w_coh.val)
    poses, vels = update_boids(poses, vels, CELL,Grid_arr,SIM,slider_dict['prot_range'].val,
                               slider_dict['vis_range'].val,slider_dict['max_speed'].val,slider_dict['dt'].val,
                               slider_dict['w_sep'].val,slider_dict['w_ali'].val,slider_dict['w_coh'].val)
    
    sc._offsets3d = (poses[:, 0], poses[:, 1], poses[:, 2])
    return (sc,)

if DIM ==2:
    ani = animation.FuncAnimation(fig, frame2d, interval=30, blit=True)

elif DIM == 3:
    ani = animation.FuncAnimation(fig, frame3d, interval=30, blit=False)
plt.show()