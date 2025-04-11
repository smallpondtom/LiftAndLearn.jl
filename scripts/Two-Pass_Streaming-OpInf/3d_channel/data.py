#%%
import glob
import re

import numpy as np
import yt

#%%
lexsort = lambda s: [
    int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)
]

#%% Data location and input parameters
path = "../../../../../DATA/NREL/3D_CHANNEL/plt*"

#%% Get list of files, and sort them
files = glob.glob(path)
files.sort(key=lexsort)

#%% Keep only the last 200 files (steady state data)
files = files[-200:]

#%% Number of time steps
nt = len(files)
Ts = np.zeros(nt)

#%% Load file initial file
f = files[0]
ds = yt.load(f)

#%% Read the first step, to setup parameters
plo = ds.domain_left_edge.d
phi = ds.domain_right_edge.d
Lx, Ly, Lz = phi - plo
nx, ny, nz = ds.domain_dimensions
x = np.linspace(0, Lx, nx) + 0.5 * Lx / nx
y = np.linspace(0, Ly, ny) + 0.5 * Ly / ny
z = np.linspace(0, Lz, nz) + 0.5 * Lz / nz

#%% Setup numpy arrays to read data into
Us = np.empty((nx, ny, nz, nt))
Vs = np.empty((nx, ny, nz, nt))
Ws = np.empty((nx, ny, nz, nt))

#%% Analytical mean
kappa = 0.384
B = 4.27
nu = 8e-6
ut = 0.0415
def log_law(u_tau, zf, nu):
    return u_tau * (np.log(zf * u_tau / nu) / kappa + B)

#%%
fields = ["velocityx", "velocityy", "velocityz"]
def read_single_snapshot(f, fields=fields, dims=ds.domain_dimensions, level=ds.max_level):
    try:
        ds = yt.load(f)
        T = float(ds.current_time)
        data = ds.covering_grid(
            left_edge=plo, fields=fields, dims=ds.domain_dimensions, level=ds.max_level
        )
        U = np.array(data["velocityx"][:])
        V = np.array(data["velocityy"][:])
        W = np.array(data["velocityz"][:])
    except:
        print(f"Failed to read {f}")
        U = np.nan
        V = np.nan
        W = np.nan
        T = np.nan
    
    return (U,V,W,T)

#%% Read all the data, load into numpy arrays
failure_count = 0
for i, f in enumerate(files):
    U,V,W,T = read_single_snapshot(f)
    Ts[i] = T
    
    if np.isnan(T):
        failure_count += 1
    else:
        Us[:, :, :, i] = U
        Vs[:, :, :, i] = V
        Ws[:, :, :, i] = W

#%%
if __name__ == "__main__":
  print(ds.domain_dimensions)