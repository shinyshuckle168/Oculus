import numpy as np
from numpy.lib.shape_base import _put_along_axis_dispatcher
from scipy.spatial.transform import Rotation as R
from fov_vect import *

positions = np.load("ECI_positions.npy").T
t = np.load("timesteps.npy")
quats = np.load("rotation_quats.npy")
rot_mat = R.from_quat(quats).as_matrix()
rot_mat = np.transpose(rot_mat, (1,2,0))
windows = np.load("imaging_windows.npy", allow_pickle=True)

# splitting data into list of passes 
pos_ls = [None]*(windows.shape[0])
t_ls = [None]*(windows.shape[0])
rot_mat_ls = [None]*windows.shape[0]

# velocity vector
v = [None]*windows.shape[0]

for i, window in enumerate(windows):
    # initial time and position if they are not provided
    if i == 0 and window[0][0]:
        # uses time step between t[0] and t[1] for initial time
        t_init = t[0] - (t[1]-t[0])
        # use velocity at t[1]
        v[0] = (positions[:,0] - positions[:,2])/(t[2] - t[0])
        # calculates initial position from velocity and time
        pos_init = positions[:,0] - v[0]*t_init
        
        t_ls[0] = np.insert(t[np.arange(window[0][0], window[0][-1]+1)], 0, t_init, axis=0)
        pos_ls[0] = np.insert(positions[:, np.arange(window[0][0], window[0][-1]+1)], 0, pos_init, axis=1)
    
    # final time and position if they are not provided
    elif i == windows.shape[1]-1 and window[0][-1] == positions.shape[2]:
        # uses time step between t[-1] and t[-2] for initial time
        t_f = t[-1] + (t[-1]-t[-2])
        # use velocity at t[-2]
        v[-1] = (positions[:,-1] - positions[:,-3])/(t[-1] - t[-3])
        # calculates initial position from velocity and time
        pos_f = positions[:,-1] + v[-1]*t_f
        
        t_ls[-1] = np.insert(t[np.arange(window[0][0], window[0][-1]+1)], 0, t_f, axis=0)
        pos_ls[-1] = np.insert(positions[:, np.arange(window[0][0], window[0][-1]+1)], 0, pos_f, axis=1)

    else:
        pos_ls[i] = positions[:, np.arange(window[0][0]-1, window[0][-1]+1)]
        t_ls[i] = t[np.arange(window[0][0]-1, window[0][-1]+1)]

    rot_mat_ls[i] = rot_mat[:, :, np.arange(window[0][0], window[0][-1])]

# Hill frame
# radial is just position since it is in ECI frame

# velocity vector
v = [None]*windows.shape[0]
for i in range(len(windows)):
    v[i] = (pos_ls[i][:,2:]-pos_ls[i][:,:-2])/2/(t_ls[i][2:]-t_ls[i][:-2])

# velocity cross radial vector
cross = [None]*windows.shape[0]
for i in range(len(windows)):
    cross[i] = np.cross(v[0], pos_ls[0][:,1:-1], axisa=0, axisb=0)

# body frame
# along track vector when at nadir
y = [None]*windows.shape[0]
for i in range(len(windows)):
    y[i] = rot_mat_ls[i][:,1,:]

# across track vector when at nadir
x = [None]*windows.shape[0]
for i in range(len(windows)):
    x[i] = rot_mat_ls[i][:,0,:]

# points to scanline
z = [None]*windows.shape[0]
for i in range(len(windows)):
    z[i] = rot_mat_ls[i][:,2,:]    

# calculating off nadir angles
# projecting radial to y-z plane; angle between projection and z is along track angle
proj_yz = [None]*windows.shape[0]
along = [None]*windows.shape[0]
for i in range(len(windows)):
    proj_yz[i] = pos_ls[i][:,1:-1] - np.einsum('ij,ij->j',pos_ls[i][:,1:-1], x[i])*x[i]
    along[i] = np.arccos(np.einsum('ij,ij->j', proj_yz[i], z[i])/np.linalg.norm(proj_yz[i], axis=0))

# projecting radial to x-z plane; angle between projection and z is across track angle
proj_xz = [None]*windows.shape[0]
across = [None]*windows.shape[0]
for i in range(len(windows)):
    proj_xz[i] = pos_ls[i][:,1:-1] - np.einsum('ij,ij->j',pos_ls[i][:,1:-1], y[i])*y[i]
    across[i] = np.arccos(np.einsum('ij,ij->j', proj_xz[i], z[i])/np.linalg.norm(proj_xz[i], axis=0))

get_px_area(across[0][:9], along[0][:9], [fov_across, fov_along], 640, pos_ls[0][:,:9], rot_mat_ls[0][:,:,:9])
