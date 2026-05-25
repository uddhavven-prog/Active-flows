import numpy as np
import numba as nb
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
)

DIM = 2
N     = 20#900
SIM   = 10*150.0
CELL  = 150.0
DT    = 10**(0)

VISUAL_RANGE    = 75.0   # for alignment + cohesion
PROTECTED_RANGE = 15.0   # for separation only

Grid_arr = np.ones(DIM,dtype=int)*int(SIM/CELL)




poses = np.random.rand(N, DIM) * SIM
angles = np.random.rand(N) * 2 * np.pi
vels  = np.stack([np.cos(angles), np.sin(angles)], axis=1) * params['max_speed'] * 0.5
vels = np.random.rand(N, DIM)
vels = vels / np.linalg.norm(vels, axis=1, keepdims=True) * params['max_speed']
# --- numba kernels (same as before) ---

import numpy as np
from numba import njit

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

'''
@nb.njit(cache=True)
def build_grid_nd(poses, cell_size, grid_dims):
    """
    poses:     2D array of shape (n, D) — particle positions
    cell_size: float — size of each grid cell
    grid_dims: 1D array of ints, shape (D,) — number of cells per dimension
    """
    n = poses.shape[0]
    D = grid_dims.shape[0]

    head = np.full(tuple(grid_dims), -1, dtype=nb.int32)
    nxt  = np.full(n,                -1, dtype=nb.int32)

    head_index = np.empty(D, dtype=nb.int64)

    for i in range(n):
        for j in range(D):
            head_index[j] = int(poses[i, j] / cell_size) % grid_dims[j]

        nxt[i]                   = head[tuple(head_index)]
        head[tuple(head_index)]  = i

    return head, nxt


@nb.njit(cache=True)
def update_boids(poses, vels, cell_size, grid_arr,
                 sim, prot_range, vis_range, max_speed, dt, w_sep, w_ali, w_coh):
    n   = poses.shape[0]
    dim = poses.shape[1]  # derive from poses, not grid_arr
    head, nxt = build_grid_nd(poses, cell_size, grid_arr)
    new_vels  = vels.copy()
    offsets   = NN_points_vec(dim)

    head_index = np.empty(dim, dtype=nb.int64)  # fix 1: dim not D

    for i in range(n):
        cell_coord = np.empty(dim, dtype=nb.int64)
        for j in range(dim):
            cell_coord[j] = int(poses[i, j] / cell_size) % grid_arr[j]

        sep   = np.zeros(dim)
        ali   = np.zeros(dim)
        coh   = np.zeros(dim)
        count = 0

        for oi in range(offsets.shape[0]):
            for k in range(dim):
                head_index[k] = (cell_coord[k] + offsets[oi, k]) % grid_arr[k]

            j = head[tuple(head_index)]  # fix 2: tuple() for N-D indexing
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

        spd = 0.0
        for k in range(dim):
            spd += new_vels[i, k] ** 2
        spd = spd ** 0.5

        if spd > max_speed:
            new_vels[i] = new_vels[i] / spd * max_speed

    new_poses = (poses + new_vels * dt) % sim
    return new_poses, new_vels

'''
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
def update_boids(poses, vels, cell_size, grid_arr,
                 sim, prot_range, vis_range, max_speed, dt, w_sep, w_ali, w_coh):
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

def setup_figure(dim, poses, vels, sim, slider_specs, params, cell_size):
    n_sliders = len(slider_specs)
    slider_height = n_sliders * 0.05 + 0.05

    fig = plt.figure(figsize=(7, 8))
    fig.subplots_adjust(bottom=slider_height + 0.05)
    fig.patch.set_facecolor("#35A704")

    speed = np.linalg.norm(vels, axis=1, keepdims=True)
    unit_vels = vels / (speed + 1e-9)

    if dim == 1:
        ax = fig.add_subplot(111)
        ax.set_xlim(0, sim)
        ax.set_ylim(-1, 1)
        ax.set_facecolor("#cbcbf9")
        ax.yaxis.set_visible(False)
        y = np.zeros(len(poses))
        artist = ax.quiver(
            poses[:, 0], y, unit_vels[:, 0], np.zeros(len(poses)),
            color='white', scale=10, headwidth=6, headlength=8, headaxislength=8
        )

    elif dim == 2:
        ax = fig.add_subplot(111)
        ax.set_xlim(0, sim)
        ax.set_ylim(0, sim)
        ax.set_aspect('equal')
        ax.set_facecolor("#cbcbf9")
        artist = ax.quiver(
            poses[:, 0], poses[:, 1], unit_vels[:, 0], unit_vels[:, 1],
            color='white', scale=100, headwidth=10, headlength=10, headaxislength=10
        )

    else:  # 3D
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(0, sim)
        ax.set_ylim(0, sim)
        ax.set_zlim(0, sim)
        ax.set_facecolor("#cbcbf9")
        artist = ax.quiver(
            poses[:, 0], poses[:, 1], poses[:, 2],
            unit_vels[:, 0], unit_vels[:, 1], unit_vels[:, 2],
            color='white', length=sim * 0.03, normalize=False
        )

    return fig, ax, artist


def setup_sliders(fig, slider_specs, params):
    n_sliders = len(slider_specs)
    slider_height = n_sliders * 0.05 + 0.05
    sliders = {}
    for i, (label, key, vmin, vmax, vinit) in enumerate(slider_specs):
        #ax_s = fig.add_axes([0.10, slider_height - i * 0.05, 0.75, 0.03])
        ax_s = fig.add_axes([0.10, 0.04 + i * 0.05, 0.75, 0.03])
        s = Slider(ax_s, label, vmin, vmax, valinit=vinit, color='steelblue')
        sliders[key] = s

    def on_change(val):
        for key, s in sliders.items():
            params[key] = s.val

    for s in sliders.values():
        s.on_changed(on_change)

    return sliders


def animate_boids(poses, vels, params, update_fn, cell_size, grid_arr, sim, dt,
                  slider_specs=None):
    dim = poses.shape[1]
    assert dim in (1, 2, 3), "Only 1D, 2D and 3D supported"

    if slider_specs is None:
        slider_specs = [
            ('separation', 'w_sep',      0.0,  0.1,       params.get('w_sep',      0.05)),
            ('alignment',  'w_ali',      0.0,  0.1,       params.get('w_ali',      0.05)),
            ('cohesion',   'w_coh',      0.0,  0.01,      params.get('w_coh',      0.005)),
            ('max speed',  'max_speed',  1.0,  40.0,      params.get('max_speed',  15.0)),
            ('prot_range', 'prot_range', 1.0,  cell_size, params.get('prot_range', 10.0)),
            ('vis_range',  'vis_range',  1.0,  cell_size, params.get('vis_range',  30.0)),
        ]

    fig, ax, artist = setup_figure(dim, poses, vels, sim, slider_specs, params, cell_size)
    setup_sliders(fig, slider_specs, params)

    def animate(frame):
        nonlocal poses, vels, artist
        poses, vels = update_fn(
            poses, vels, cell_size, grid_arr, sim,
            params['prot_range'], params['vis_range'], params['max_speed'], dt,
            params['w_sep'], params['w_ali'], params['w_coh']
        )

        speed = np.linalg.norm(vels, axis=1, keepdims=True)
        unit_vels = vels / (speed + 1e-9)

        if dim == 1:
            artist.set_offsets(np.c_[poses[:, 0], np.zeros(len(poses))])
            artist.set_UVC(unit_vels[:, 0], np.zeros(len(poses)))
        elif dim == 2:
            artist.set_offsets(poses)
            artist.set_UVC(unit_vels[:, 0], unit_vels[:, 1])
        else:
            artist.remove()
            artist = ax.quiver(
                poses[:, 0], poses[:, 1], poses[:, 2],
                unit_vels[:, 0], unit_vels[:, 1], unit_vels[:, 2],
                color='white', length=sim * 0.03, normalize=False
            )
        return artist,

    ani = animation.FuncAnimation(fig, animate, interval=16, blit=dim != 3)
    plt.show()
    return ani


def animate_boids(poses, vels, params, update_fn, cell_size, grid_arr, sim, dt,
                  slider_specs=None):
    dim = poses.shape[1]
    assert dim in (1, 2, 3), "Only 1D, 2D and 3D supported"

    if slider_specs is None:
        slider_specs = [
            ('separation', 'w_sep',      0.0,  0.1,       params.get('w_sep',      0.05)),
            ('alignment',  'w_ali',      0.0,  0.1,       params.get('w_ali',      0.05)),
            ('cohesion',   'w_coh',      0.0,  0.01,      params.get('w_coh',      0.005)),
            ('max speed',  'max_speed',  1.0,  40.0,      params.get('max_speed',  15.0)),
            ('prot_range', 'prot_range', 1.0,  cell_size, params.get('prot_range', 10.0)),
            ('vis_range',  'vis_range',  1.0,  cell_size, params.get('vis_range',  30.0)),
        ]

    fig, ax, artist = setup_figure(dim, poses, vels, sim, slider_specs, params, cell_size)
    setup_sliders(fig, slider_specs, params)

    def animate1(frame):
        nonlocal poses, vels, artist
        poses, vels = update_fn(
            poses, vels, cell_size, grid_arr, sim,
            params['prot_range'], params['vis_range'], params['max_speed'], dt,
            params['w_sep'], params['w_ali'], params['w_coh']
        )

        speed = np.linalg.norm(vels, axis=1, keepdims=True)
        unit_vels = vels / (speed + 1e-9)

        artist.set_offsets(np.c_[poses[:, 0], np.zeros(len(poses))])
        artist.set_UVC(unit_vels[:, 0], np.zeros(len(poses)))
        return artist,

    def animate2(frame):
        nonlocal poses, vels, artist
        poses, vels = update_fn(
            poses, vels, cell_size, grid_arr, sim,
            params['prot_range'], params['vis_range'], params['max_speed'], dt,
            params['w_sep'], params['w_ali'], params['w_coh']
        )
        speed = np.linalg.norm(vels, axis=1, keepdims=True)
        unit_vels = vels / (speed + 1e-9)

        
        artist.set_offsets(poses)
        artist.set_UVC(unit_vels[:, 0], unit_vels[:, 1])
        
        return artist,
    def animate3(frame):
        nonlocal poses, vels, artist
        poses, vels = update_fn(
            poses, vels, cell_size, grid_arr, sim,
            params['prot_range'], params['vis_range'], params['max_speed'], dt,
            params['w_sep'], params['w_ali'], params['w_coh']
        )

        speed = np.linalg.norm(vels, axis=1, keepdims=True)
        unit_vels = vels / (speed + 1e-9)

        
        artist.remove()
        artist = ax.quiver(
            poses[:, 0], poses[:, 1], poses[:, 2],
            unit_vels[:, 0], unit_vels[:, 1], unit_vels[:, 2],
            color='white', length=sim * 0.03, normalize=False
        )
        return artist,
    
    if dim ==1:
        animate = animate1
    elif dim ==2:
        animate = animate2
    else:
        animate = animate3
    
    ani = animation.FuncAnimation(fig, animate, interval=16, blit=False)
    plt.show()
    return ani
#ani = animate_boids(poses, vels, params, update_boids, CELL, Grid_arr, SIM, DT)


'''


def animate_boids(poses, vels, params, update_fn, cell_size, grid_arr, sim, dt,
                  slider_specs=None):
    dim = poses.shape[1]
    assert dim in (1, 2, 3), "Only 1D, 2D and 3D supported"

    if slider_specs is None:
        slider_specs = [
            ('separation', 'w_sep',      0.0,  0.1,       params.get('w_sep',      0.05)),
            ('alignment',  'w_ali',      0.0,  0.1,       params.get('w_ali',      0.05)),
            ('cohesion',   'w_coh',      0.0,  0.01,      params.get('w_coh',      0.005)),
            ('max speed',  'max_speed',  1.0,  40.0,      params.get('max_speed',  15.0)),
            ('prot_range', 'prot_range', 1.0,  cell_size, params.get('prot_range', 10.0)),
            ('vis_range',  'vis_range',  1.0,  cell_size, params.get('vis_range',  30.0)),
        ]

    fig, ax, artist = setup_figure(dim, poses, vels, sim, slider_specs, params, cell_size)
    setup_sliders(fig, slider_specs, params)

    def step():
        nonlocal poses, vels, artist
        poses, vels = update_fn(
            poses, vels, cell_size, grid_arr, sim,
            params['prot_range'], params['vis_range'], params['max_speed'], dt,
            params['w_sep'], params['w_ali'], params['w_coh']
        )
        speed = np.linalg.norm(vels, axis=1, keepdims=True)
        unit_vels = vels / (speed + 1e-9)

        if dim == 1:
            artist.set_offsets(np.c_[poses[:, 0], np.zeros(len(poses))])
            artist.set_UVC(unit_vels[:, 0], np.zeros(len(poses)))
        elif dim == 2:
            artist.set_offsets(poses)
            artist.set_UVC(unit_vels[:, 0], unit_vels[:, 1])
        else:
            artist.remove()
            artist = ax.quiver(
                poses[:, 0], poses[:, 1], poses[:, 2],
                unit_vels[:, 0], unit_vels[:, 1], unit_vels[:, 2],
                color='white', length=sim * 0.03, normalize=False
            )
        fig.canvas.draw_idle()

    timer = fig.canvas.new_timer(interval=16)
    timer.add_callback(step)
    timer.start()

    plt.show()
    return timer  # must keep reference or GC kills it


ani = animate_boids(poses, vels, params, update_boids, CELL, Grid_arr, SIM, DT)
'''

def animate_boids(poses, vels, params, update_fn, cell_size, grid_arr, sim, dt,
                  slider_specs=None):
    dim = poses.shape[1]
    assert dim in (1, 2, 3), "Only 1D, 2D and 3D supported"

    if slider_specs is None:
        slider_specs = [
            ('separation', 'w_sep',      0.0,  0.1,       params.get('w_sep',      0.05)),
            ('alignment',  'w_ali',      0.0,  0.1,       params.get('w_ali',      0.05)),
            ('cohesion',   'w_coh',      0.0,  0.01,      params.get('w_coh',      0.005)),
            ('max speed',  'max_speed',  1.0,  40.0,      params.get('max_speed',  15.0)),
            ('prot_range', 'prot_range', 1.0,  cell_size, params.get('prot_range', 10.0)),
            ('vis_range',  'vis_range',  1.0,  cell_size, params.get('vis_range',  30.0)),
        ]

    fig, ax, artist = setup_figure(dim, poses, vels, sim, slider_specs, params, cell_size)
    setup_sliders(fig, slider_specs, params)

    # grab the underlying Tk root window
    root = fig.canvas.get_tk_widget().winfo_toplevel()

    def step():
        nonlocal poses, vels, artist
        poses, vels = update_fn(
            poses, vels, cell_size, grid_arr, sim,
            params['prot_range'], params['vis_range'], params['max_speed'], dt,
            params['w_sep'], params['w_ali'], params['w_coh']
        )
        speed = np.linalg.norm(vels, axis=1, keepdims=True)
        unit_vels = vels / (speed + 1e-9)

        if dim == 1:
            artist.set_offsets(np.c_[poses[:, 0], np.zeros(len(poses))])
            artist.set_UVC(unit_vels[:, 0], np.zeros(len(poses)))
        elif dim == 2:
            artist.set_offsets(poses)
            artist.set_UVC(unit_vels[:, 0], unit_vels[:, 1])
        else:
            artist.remove()
            artist = ax.quiver(
                poses[:, 0], poses[:, 1], poses[:, 2],
                unit_vels[:, 0], unit_vels[:, 1], unit_vels[:, 2],
                color='white', length=sim * 0.03, normalize=False
            )

        fig.canvas.draw_idle()
        root.after(16, step)  # schedule next frame via Tk — never blocks the event loop

    root.after(16, step)
    plt.show()  # hands control to Tk's mainloop


ani = animate_boids(poses, vels, params, update_boids, CELL, Grid_arr, SIM, DT)