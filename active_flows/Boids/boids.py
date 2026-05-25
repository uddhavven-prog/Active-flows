import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider

# --- parameters (mutable dict so the animation closure sees updates) ---
params = dict(
    w_sep      = 0.05,
    w_ali      = 0.05,
    w_coh      = 0.005,
    vis_range  = 75.0,   # was 'radius', now split into two
    prot_range = 15.0,
    max_speed  = 15.0,
)
N     = 900
SIM   = 3*150.0
CELL  = 150.0
DT    = 10**(0)

VISUAL_RANGE    = 75.0   # for alignment + cohesion
PROTECTED_RANGE = 15.0   # for separation only


grid_w = int(SIM / CELL)
grid_h = int(SIM / CELL)



poses = np.random.rand(N, 2) * SIM
angles = np.random.rand(N) * 2 * np.pi
vels  = np.stack([np.cos(angles), np.sin(angles)], axis=1) * params['max_speed'] * 0.5

# --- numba kernels (same as before) ---

@nb.njit(cache=True)
def build_grid(poses, cell_size, grid_w, grid_h):
    n = poses.shape[0]
    n_cells = grid_w * grid_h
    head = np.full(n_cells, -1, dtype=nb.int32)
    nxt  = np.full(n,       -1, dtype=nb.int32)
    for i in range(n):
        cx = int(poses[i, 0] / cell_size) % grid_w
        cy = int(poses[i, 1] / cell_size) % grid_h
        cell = cy * grid_w + cx
        nxt[i]     = head[cell]
        head[cell] = i
    return head, nxt

@nb.njit(cache=True)
def update_boids(poses, vels, cell_size, grid_w, grid_h,
                 sim, prot_range, vis_range, max_speed, dt, w_sep, w_ali, w_coh):
    n = poses.shape[0]
    head, nxt = build_grid(poses, cell_size, grid_w, grid_h)
    new_vels = vels.copy()

    for i in range(n):
        cx = int(poses[i, 0] / cell_size) % grid_w
        cy = int(poses[i, 1] / cell_size) % grid_h

        sep_x = sep_y = 0.0
        ali_x = ali_y = 0.0
        coh_x = coh_y = 0.0
        count = 0

        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx_ = (cx + dx) % grid_w
                ny_ = (cy + dy) % grid_h
                cell = ny_ * grid_w + nx_
                j = head[cell]
                while j != -1:
                    if j != i:
                        ddx = poses[j, 0] - poses[i, 0]
                        ddy = poses[j, 1] - poses[i, 1]
                        if ddx >  sim * 0.5: ddx -= sim
                        if ddx < -sim * 0.5: ddx += sim
                        if ddy >  sim * 0.5: ddy -= sim
                        if ddy < -sim * 0.5: ddy += sim
                        dist = (ddx*ddx + ddy*ddy) ** 0.5
                        '''
                        if dist < radius:
                            sep_x -= ddx / (dist + 1e-9)
                            sep_y -= ddy / (dist + 1e-9)
                            ali_x += vels[j, 0]
                            ali_y += vels[j, 1]
                            coh_x += ddx
                            coh_y += ddy
                            count += 1
                        '''
                        if dist < prot_range:
                            sep_x -= ddx
                            sep_y -= ddy
                        elif dist < vis_range:
                            ali_x += vels[j, 0]
                            ali_y += vels[j, 1]
                            coh_x += ddx
                            coh_y += ddy
                            count += 1
                    j = nxt[j]

        if count > 0:
            ali_x = ali_x / count - vels[i, 0]
            ali_y = ali_y / count - vels[i, 1]
            coh_x /= count
            coh_y /= count

        sx = w_sep*sep_x + w_ali*ali_x + w_coh*coh_x
        sy = w_sep*sep_y + w_ali*ali_y + w_coh*coh_y

        new_vels[i, 0] = vels[i, 0] + sx * dt
        new_vels[i, 1] = vels[i, 1] + sy * dt

        spd = (new_vels[i,0]**2 + new_vels[i,1]**2) ** 0.5
        if spd > max_speed:
            new_vels[i, 0] = new_vels[i, 0] / spd * max_speed
            new_vels[i, 1] = new_vels[i, 1] / spd * max_speed

    new_poses = (poses + new_vels * dt) % sim
    return new_poses, new_vels

# --- warmup ---
poses, vels = update_boids(poses, vels, CELL, grid_w, grid_h, SIM,
                            params['prot_range'],params['vis_range'],
                              params['max_speed'], DT,
                            params['w_sep'], params['w_ali'], params['w_coh'])

# --- figure layout ---
fig, ax = plt.subplots(figsize=(7, 8))
fig.subplots_adjust(bottom=0.35)  # leave room for sliders
ax.set_xlim(0, SIM)
ax.set_ylim(0, SIM)
ax.set_aspect('equal')
ax.set_facecolor("#cbcbf9")
fig.patch.set_facecolor("#35A704")

#scat = ax.scatter(poses[:, 0], poses[:, 1], s=4, c='white', linewidths=0)
scat = ax.quiver(poses[:, 0], poses[:, 1], vels[:, 0], vels[:, 1],
                 color='white', scale=100, headwidth=10, headlength=10, headaxislength=10)

# --- sliders ---

w_sep_init     = 0.05
w_ali_init     = 0.05
w_coh_init     = 0.005
radius_init    = 75.0
max_speed_init = 15.0



slider_specs = [
    #  (label,       param key,    left,  min,  max,  init)
    ('separation',  'w_sep',      0.10,  0.0,  0.1,  w_sep_init),
    ('alignment',   'w_ali',      0.10,  0.0,  0.1,  w_ali_init),
    ('cohesion',    'w_coh',      0.10,  0.0,  0.01,  w_coh_init),
    #('radius',      'radius',     0.10,  1.0, CELL, radius_init),
    ('max speed',   'max_speed',  0.10,  1.0, 40.0, max_speed_init),
    ('prot_range', 'prot_range', 0.10,  1.0, CELL, params['prot_range']),
    ('vis_range',  'vis_range',  0.10,  1.0, CELL, params['vis_range']),
]

sliders = {}
for i, (label, key, left, vmin, vmax, vinit) in enumerate(slider_specs):
    ax_s = fig.add_axes([left, 0.28 - i*0.05, 0.75, 0.03])
    s = Slider(ax_s, label, vmin, vmax, valinit=vinit, color='steelblue')
    sliders[key] = s

def on_change(val):
    for key, s in sliders.items():
        params[key] = s.val

for s in sliders.values():
    s.on_changed(on_change)

# --- animate ---


def animate(frame):
    global poses, vels
    poses, vels = update_boids(
        poses, vels, CELL, grid_w, grid_h, SIM,
        params['prot_range'], params['vis_range'], params['max_speed'], DT,
        params['w_sep'], params['w_ali'], params['w_coh']
    )
    speed = np.linalg.norm(vels, axis=1, keepdims=True)
    unit_vels = vels / (speed + 1e-9)

    scat.set_offsets(poses)
    scat.set_UVC(unit_vels[:, 0], unit_vels[:, 1])
    return scat,
ani = animation.FuncAnimation(fig, animate, interval=16, blit=True)
plt.show()